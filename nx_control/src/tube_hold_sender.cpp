#include "nx_control/io.hpp"
#include "nx_control/protocol.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kDefaultStartCommandId = 1U;
constexpr std::uint64_t kReservationSize = 4096U;
constexpr auto kPeriod = std::chrono::milliseconds(20);
constexpr std::size_t kStateRecordSize = 21U;

std::atomic<bool> stopping{false};

void signal_handler(int) { stopping.store(true); }

std::uint64_t parse_u64(const std::string& value, const std::string& option) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument("invalid value for " + option);
  }
  std::size_t consumed = 0;
  const unsigned long long result = std::stoull(value, &consumed, 0);
  if (consumed != value.size()) throw std::invalid_argument("invalid value for " + option);
  return static_cast<std::uint64_t>(result);
}

struct Options {
  std::string dmmc = "/dev/ttyACM0";
  int baud = 921600;
  std::string state_file = "tube-hold-v3.seq";
  std::uint32_t start_command_id = kDefaultStartCommandId;
  std::uint64_t max_frames = 0;
  bool clear_faults_on_first_frame = true;
  bool dry_run = false;
  bool verbose = false;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) throw std::invalid_argument("missing value after " + argument);
      return argv[index];
    };
    if (argument == "--dmmc") {
      options.dmmc = value();
    } else if (argument == "--baud") {
      const std::uint64_t parsed = parse_u64(value(), argument);
      if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("--baud exceeds int");
      }
      options.baud = static_cast<int>(parsed);
    } else if (argument == "--state-file") {
      options.state_file = value();
    } else if (argument == "--start-command-id") {
      const std::uint64_t parsed = parse_u64(value(), argument);
      if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--start-command-id exceeds uint32");
      }
      options.start_command_id = static_cast<std::uint32_t>(parsed);
    } else if (argument == "--max-frames") {
      options.max_frames = parse_u64(value(), argument);
    } else if (argument == "--no-clear-faults") {
      options.clear_faults_on_first_frame = false;
    } else if (argument == "--dry-run") {
      options.dry_run = true;
    } else if (argument == "--verbose") {
      options.verbose = true;
    } else if (argument == "--help") {
      std::cout
          << "tube_hold_sender [--dmmc DEVICE] [--baud RATE]\n"
             "  [--state-file FILE] [--start-command-id ID] [--no-clear-faults]\n"
             "  [--max-frames N] [--dry-run] [--verbose]\n\n"
             "Continuously sends tube-control-v3 HOLD frames at 50 Hz.  The first\n"
             "frame uses flags=0x03 by default; later frames use flags=0x01.\n"
             "The sequence file must be kept across restarts and must not be shared\n"
             "between different senders.  --start-command-id is used only when a\n"
             "new/empty sequence file is created.\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.state_file.empty() && !options.dry_run) {
    throw std::invalid_argument("--state-file cannot be empty");
  }
  return options;
}

class PersistentSequence {
 public:
  PersistentSequence(const std::string& path, std::uint32_t initial) : path_(path) {
    const std::filesystem::path state_path(path);
    if (!state_path.parent_path().empty()) {
      std::filesystem::create_directories(state_path.parent_path());
    }
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd_ < 0) throw_system_error("cannot open sequence file");
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      const int saved_errno = errno;
      ::close(fd_);
      fd_ = -1;
      errno = saved_errno;
      throw_system_error("sequence file is already locked by another sender");
    }
    next_unreserved_ = read_state(initial);
    next_ = next_unreserved_;
    reserved_end_ = next_unreserved_;
  }

  ~PersistentSequence() {
    if (fd_ >= 0) ::close(fd_);
  }

  PersistentSequence(const PersistentSequence&) = delete;
  PersistentSequence& operator=(const PersistentSequence&) = delete;

  void prepare() {
    if (next_ == reserved_end_) reserve();
  }

  std::uint32_t next() {
    if (next_ == reserved_end_) reserve();
    const std::uint64_t value = next_++;
    return static_cast<std::uint32_t>(value);
  }

  // On a clean stop, retain the exact next ID.  On an abrupt stop, the
  // pre-synced reservation watermark remains, so possibly transmitted IDs are
  // skipped rather than reused.
  void checkpoint() { persist(next_); }

 private:
  [[noreturn]] void throw_system_error(const std::string& message) const {
    throw std::runtime_error(message + " '" + path_ + "': " + std::strerror(errno));
  }

  std::uint64_t read_state(std::uint32_t initial) {
    char record[kStateRecordSize + 1U];
    const ssize_t count = ::pread(fd_, record, sizeof(record), 0);
    if (count < 0) throw_system_error("cannot read sequence file");
    if (count == 0) return initial;
    if (count != static_cast<ssize_t>(kStateRecordSize) || record[20] != '\n') {
      throw std::runtime_error("invalid sequence file '" + path_ +
                               "'; refusing to risk an ID reuse");
    }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 20U; ++index) {
      if (record[index] < '0' || record[index] > '9') {
        throw std::runtime_error("invalid sequence file '" + path_ +
                                 "'; refusing to risk an ID reuse");
      }
      value = value * 10U + static_cast<unsigned>(record[index] - '0');
    }
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1U) {
      throw std::runtime_error("sequence file value is out of range");
    }
    return value;
  }

  void reserve() {
    constexpr std::uint64_t kEnd =
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1U;
    if (next_unreserved_ >= kEnd) {
      throw std::runtime_error("command_id space exhausted; refusing to wrap or repeat");
    }
    next_ = next_unreserved_;
    reserved_end_ = std::min(kEnd, next_ + kReservationSize);
    persist(reserved_end_);
    next_unreserved_ = reserved_end_;
  }

  void persist(std::uint64_t value) {
    std::ostringstream stream;
    stream << std::setw(20) << std::setfill('0') << value << '\n';
    const std::string record = stream.str();
    const ssize_t count = ::pwrite(fd_, record.data(), record.size(), 0);
    if (count != static_cast<ssize_t>(record.size())) {
      if (count >= 0) errno = EIO;
      throw_system_error("cannot update sequence file");
    }
    if (::fdatasync(fd_) != 0) throw_system_error("cannot sync sequence file");
  }

  std::string path_;
  int fd_ = -1;
  std::uint64_t next_ = 0;
  std::uint64_t reserved_end_ = 0;
  std::uint64_t next_unreserved_ = 0;
};

std::uint32_t nx_time_ms() {
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch());
  return static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(milliseconds.count()) & 0xFFFFFFFFU);
}

std::string hex_packet(const std::vector<std::uint8_t>& packet) {
  std::ostringstream stream;
  stream << std::hex << std::uppercase << std::setfill('0');
  for (std::size_t index = 0; index < packet.size(); ++index) {
    if (index != 0) stream << ' ';
    stream << std::setw(2) << static_cast<unsigned>(packet[index]);
  }
  return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::unique_ptr<PersistentSequence> persistent_sequence;
    if (!options.dry_run) {
      persistent_sequence =
          std::make_unique<PersistentSequence>(options.state_file, options.start_command_id);
    }
    nx_control::SerialPort serial;
    if (!options.dry_run && !serial.open(options.dmmc, options.baud)) {
      throw std::runtime_error("cannot open DMMC serial '" + options.dmmc +
                               "': " + serial.last_error());
    }
    if (persistent_sequence) persistent_sequence->prepare();
    std::uint64_t dry_sequence = options.start_command_id;
    std::uint64_t sent_frames = 0;

    std::cerr << (options.dry_run ? "dry-run" : "sending") << " tube-control-v3 at 50 Hz"
              << " state=HOLD(1) theta=+20cdeg rate_limit=200cdeg/s ttl=200ms"
              << " first_flags=0x"
              << (options.clear_faults_on_first_frame ? "03" : "01") << '\n';

    auto next_tick = std::chrono::steady_clock::now();
    while (!stopping.load() &&
           (options.max_frames == 0U || sent_frames < options.max_frames)) {
      std::this_thread::sleep_until(next_tick);
      if (stopping.load()) break;

      std::uint32_t id = 0;
      if (persistent_sequence) {
        id = persistent_sequence->next();
      } else {
        if (dry_sequence > std::numeric_limits<std::uint32_t>::max()) {
          throw std::runtime_error("command_id space exhausted; refusing to wrap or repeat");
        }
        id = static_cast<std::uint32_t>(dry_sequence++);
      }

      nx_control::TubeControlV3Command command;
      command.command_id = id;
      command.source_frame_id = id;
      command.nx_time_ms = nx_time_ms();
      command.theta_cmd_cdeg = 20;
      command.theta_rate_limit_cdeg_s = 200;
      command.ttl_ms = 200;
      command.control_state = 1;
      command.flags =
          sent_frames == 0U && options.clear_faults_on_first_frame ? 0x03U : 0x01U;
      command.reserved = 0;
      const std::vector<std::uint8_t> packet =
          nx_control::protocol::encode_tube_control_v3(command);

      if (!options.dry_run && !serial.write_all(packet.data(), packet.size())) {
        // The receiver may have accepted a frame even when the local write
        // reports an error.  Never retry this ID.
        throw std::runtime_error("DMMC serial write failed after command_id=" +
                                 std::to_string(id) + ": " + serial.last_error());
      }
      ++sent_frames;

      if (options.verbose) {
        std::cerr << (options.dry_run ? "frame" : "sent")
                  << " command_id=" << command.command_id
                  << " source_frame_id=" << command.source_frame_id
                  << " nx_time_ms=" << command.nx_time_ms
                  << " flags=0x" << std::hex << static_cast<unsigned>(command.flags)
                  << std::dec << " bytes=" << hex_packet(packet) << '\n';
      } else if (sent_frames % 50U == 0U) {
        std::cerr << "sent_frames=" << sent_frames << " last_command_id=" << id << '\n';
      }

      next_tick += kPeriod;
      const auto now = std::chrono::steady_clock::now();
      if (now > next_tick) next_tick = now + kPeriod;
    }

    if (persistent_sequence) persistent_sequence->checkpoint();
    std::cerr << "stopped after " << sent_frames << " frame(s)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
