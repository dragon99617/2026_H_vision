from __future__ import annotations

import cv2

from .types import TrackStatus


STATUS_NAMES = {
    TrackStatus.LOST: "lost",
    TrackStatus.MEASURED: "measured",
    TrackStatus.PREDICTED: "predicted",
}


def _draw_text_panel(
    canvas,
    text: str,
    origin,
    color,
    font_scale: float,
    thickness: int,
) -> None:
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    x = min(
        max(6, int(origin[0])),
        max(6, canvas.shape[1] - text_width - 14),
    )
    y = min(
        max(text_height + 8, int(origin[1])),
        canvas.shape[0] - baseline - 8,
    )
    cv2.rectangle(
        canvas,
        (x - 6, y - text_height - 7),
        (x + text_width + 6, y + baseline + 6),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        canvas,
        (x - 6, y - text_height - 7),
        (x + text_width + 6, y + baseline + 6),
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_tube_ruler(canvas, negative, positive, mode: str) -> None:
    negative = tuple(float(value) for value in negative)
    positive = tuple(float(value) for value in positive)
    dx = positive[0] - negative[0]
    dy = positive[1] - negative[1]
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    perpendicular = (-dy / length, dx / length)
    if mode == "rgbd":
        axis_color = (255, 0, 255)
        tick_color = (255, 255, 0)
        mode_text = "3D TUBE SCALE (cm)"
    elif mode == "rgb-contour":
        axis_color = (0, 220, 0)
        tick_color = (0, 255, 255)
        mode_text = "RGB CONTOUR SCALE (cm)"
    else:
        axis_color = (0, 165, 255)
        tick_color = (0, 210, 255)
        mode_text = "2D REFERENCE ONLY - DEPTH INVALID"
    start = (int(round(negative[0])), int(round(negative[1])))
    end = (int(round(positive[0])), int(round(positive[1])))
    cv2.line(canvas, start, end, axis_color, 3, cv2.LINE_AA)

    minor_values = [-12.5 + 2.5 * index for index in range(11)]
    major_values = {-12.5, -6.25, 0.0, 6.25, 12.5}
    values = sorted(set(minor_values).union(major_values))
    for value in values:
        fraction = (value + 12.5) / 25.0
        x = negative[0] + dx * fraction
        y = negative[1] + dy * fraction
        major = value in major_values
        half_tick = 14.0 if major else 8.0
        p1 = (
            int(round(x - perpendicular[0] * half_tick)),
            int(round(y - perpendicular[1] * half_tick)),
        )
        p2 = (
            int(round(x + perpendicular[0] * half_tick)),
            int(round(y + perpendicular[1] * half_tick)),
        )
        cv2.line(canvas, p1, p2, tick_color, 2 if major else 1, cv2.LINE_AA)
        if major:
            label = "%+g" % value if value else "0"
            cv2.putText(
                canvas,
                label,
                (p2[0] + 4, p2[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                tick_color,
                1,
                cv2.LINE_AA,
            )
    midpoint = (
        int(round((negative[0] + positive[0]) * 0.5)),
        int(round((negative[1] + positive[1]) * 0.5)),
    )
    cv2.putText(
        canvas,
        mode_text,
        (max(5, midpoint[0] - 150), max(20, midpoint[1] + 42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        axis_color,
        2,
        cv2.LINE_AA,
    )


def draw_web_preview(frame, result):
    """Draw the ball detection and tube ruler used by the web video feed."""
    canvas = frame.copy()
    if result is None:
        return canvas

    ball_status = result.effective_ball_status
    if ball_status == TrackStatus.MEASURED:
        color = (0, 220, 0)
    elif ball_status == TrackStatus.PREDICTED:
        color = (0, 210, 255)
    else:
        color = (0, 80, 255)

    detection = result.detection
    if detection is not None:
        x1 = int(round(detection.x1))
        y1 = int(round(detection.y1))
        x2 = int(round(detection.x2))
        y2 = int(round(detection.y2))
        center = tuple(int(round(value)) for value in detection.center)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=17,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        label = "BALL %.2f" % detection.confidence
        if result.position_cm is not None and result.status != TrackStatus.LOST:
            label += "  %+.2f cm" % result.position_cm
        _draw_text_panel(
            canvas,
            label,
            (x1, max(26, y1 - 8)),
            color,
            0.58,
            2,
        )

    contour = result.tube_contour
    pose = result.tube_pose
    if (
        pose is not None
        and pose.valid
        and pose.endpoint_negative_px is not None
        and pose.endpoint_positive_px is not None
    ):
        _draw_tube_ruler(
            canvas,
            pose.endpoint_negative_px,
            pose.endpoint_positive_px,
            "rgbd",
        )
    elif (
        contour is not None
        and contour.valid
        and contour.endpoint_negative_px is not None
        and contour.endpoint_positive_px is not None
    ):
        _draw_tube_ruler(
            canvas,
            contour.endpoint_negative_px,
            contour.endpoint_positive_px,
            (
                "rgb-contour"
                if result.position_source == "rgb-contour"
                else "reference"
            ),
        )

    if result.position_projected_px is not None:
        point = tuple(int(round(value)) for value in result.position_projected_px)
        if detection is not None:
            cv2.line(
                canvas,
                tuple(int(round(value)) for value in detection.center),
                point,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.drawMarker(
            canvas,
            point,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            19,
            2,
            cv2.LINE_AA,
        )
    return canvas


def draw_debug(
    frame,
    result,
    capture_frame_id: int,
    capture_fps: float,
    depth_fps: float,
    pose_fps: float,
    inference_fps: float,
    skipped_frames: int,
    latency_p95_ms: float,
    serial_text: str,
    serial_fps: float,
    engine_shape,
    fast_path: str,
    camera_device: str,
    power_mode: str,
    protocol: str,
    color_controls_text: str = "exposure unavailable",
):
    canvas = frame.copy()
    status = result.status if result is not None else TrackStatus.LOST
    ball_status = (
        result.effective_ball_status if result is not None else TrackStatus.LOST
    )
    detection = result.detection if result is not None else None
    if ball_status == TrackStatus.MEASURED:
        color = (0, 220, 0)
    elif ball_status == TrackStatus.PREDICTED:
        color = (0, 210, 255)
    else:
        color = (0, 80, 255)

    if detection is not None:
        x1 = int(round(detection.x1))
        y1 = int(round(detection.y1))
        x2 = int(round(detection.x2))
        y2 = int(round(detection.y2))
        center_x, center_y = detection.center
        center = (int(round(center_x)), int(round(center_y)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=15,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        label = "%s conf=%.3f center=(%d,%d)" % (
            STATUS_NAMES[ball_status],
            detection.confidence,
            center[0],
            center[1],
        )
        cv2.putText(
            canvas,
            label,
            (x1, max(80, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )

    contour = result.tube_contour if result is not None else None
    if contour is not None and contour.contour is not None:
        cv2.polylines(
            canvas,
            [contour.contour.astype("int32")],
            True,
            (255, 180, 0) if contour.valid else (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
    pose = result.tube_pose if result is not None else None
    if (
        pose is not None
        and pose.valid
        and pose.endpoint_negative_px is not None
        and pose.endpoint_positive_px is not None
    ):
        _draw_tube_ruler(
            canvas,
            pose.endpoint_negative_px,
            pose.endpoint_positive_px,
            "rgbd",
        )
    elif (
        contour is not None
        and contour.valid
        and contour.endpoint_negative_px is not None
        and contour.endpoint_positive_px is not None
    ):
        _draw_tube_ruler(
            canvas,
            contour.endpoint_negative_px,
            contour.endpoint_positive_px,
            (
                "rgb-contour"
                if result is not None
                and result.position_source == "rgb-contour"
                else "reference"
            ),
        )
    if result is not None and result.position_projected_px is not None:
        point = tuple(
            int(round(value)) for value in result.position_projected_px
        )
        if detection is not None:
            ball_center = tuple(
                int(round(value)) for value in detection.center
            )
            cv2.line(
                canvas,
                ball_center,
                point,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.drawMarker(
            canvas,
            point,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            19,
            2,
            cv2.LINE_AA,
        )
        if result.position_cm is not None:
            _draw_text_panel(
                canvas,
                "%+.2f cm" % result.position_cm,
                (point[0] + 16, point[1] - 16),
                color,
                0.72,
                2,
            )

    if (
        result is not None
        and result.position_cm is not None
        and result.status != TrackStatus.LOST
    ):
        position_text = "BALL POSITION: %+.2f cm" % result.position_cm
        position_color = color
    else:
        position_text = "BALL POSITION: INVALID"
        position_color = (0, 80, 255)
    _draw_text_panel(
        canvas,
        position_text,
        (18, canvas.shape[0] - 24),
        position_color,
        1.0,
        2,
    )

    result_frame_id = result.frame_id if result is not None else 0
    frame_age = max(0, capture_frame_id - result_frame_id)
    rgb_contour_mode = (
        result is not None and result.position_source == "rgb-contour"
    )
    if rgb_contour_mode:
        geometry_status = "RGB contour projection"
        geometry_detail = "tube projection %.1f px | no depth | protocol %s" % (
            contour.projected_length_px if contour is not None else 0.0,
            protocol,
        )
    else:
        geometry_status = (
            "valid"
            if pose is not None and pose.valid
            else (pose.reason if pose is not None else "unavailable")
        )
        geometry_detail = (
            "pitch %+.2f deg | fit RMS %.2f mm | depth bins %.1f%% | protocol %s"
            % (
                pose.pitch_degrees if pose is not None else 0.0,
                pose.rms_m * 1000.0 if pose is not None else 0.0,
                pose.valid_bin_ratio * 100.0 if pose is not None else 0.0,
                protocol,
            )
        )
    lines = [
        "capture %.1f FPS | depth %.1f FPS | tube %.1f FPS | infer %.1f FPS | e2e %.1f ms P95 %.1f ms"
        % (
            capture_fps,
            depth_fps,
            pose_fps,
            inference_fps,
            result.latency_ms if result is not None else 0.0,
            latency_p95_ms,
        ),
        "pre %.2f ms | gpu %.2f ms | post %.2f ms | geometry %.2f ms | total %.2f ms"
        % (
            result.preprocess_ms if result is not None else 0.0,
            result.gpu_ms if result is not None else 0.0,
            result.postprocess_ms if result is not None else 0.0,
            result.geometry_ms if result is not None else 0.0,
            result.inference_ms if result is not None else 0.0,
        ),
        "capture_frame %d | infer_frame %d | age %d | skipped %d | %s"
        % (
            capture_frame_id,
            result_frame_id,
            frame_age,
            skipped_frames,
            "position=%s ball=%s"
            % (STATUS_NAMES[status], STATUS_NAMES[ball_status]),
        ),
        "position %s | tube conf %.3f | depth age %s | geometry %s"
        % (
            (
                "%+.2f cm" % result.position_cm
                if result is not None and result.position_cm is not None
                else "INVALID"
            ),
            result.tube_confidence if result is not None else 0.0,
            (
                "n/a"
                if rgb_contour_mode
                else "%.1f ms"
                % (
                    result.depth_age_ms
                    if result is not None
                    else float("inf")
                )
            ),
            geometry_status,
        ),
        geometry_detail,
        "engine %s | %s | camera %s | %s | power %s | serial %s %.1f Hz"
        % (
            tuple(engine_shape),
            fast_path,
            camera_device,
            color_controls_text,
            power_mode,
            serial_text,
            serial_fps,
        ),
    ]
    overlay_height = 20 + len(lines) * 25
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], overlay_height), (0, 0, 0), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (12, 25 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
