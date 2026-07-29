from __future__ import annotations


def install_torchvision_nms_fallback() -> bool:
    """Use CPU NMS when NVIDIA's JetPack 5 torchvision lacks an SM72 kernel.

    YOLO26 inference is end-to-end and does not use NMS. Ultralytics 8.4.102
    nevertheless executes a tiny synthetic NMS during final-validation warmup.
    NVIDIA's prebuilt torchvision 0.16.2 wheel does not contain that CUDA
    kernel for Xavier (SM72), so keep this narrowly scoped compatibility
    fallback in the training/evaluation processes.
    """
    import torch
    import torchvision

    if not torch.cuda.is_available():
        return False
    if tuple(torch.cuda.get_device_capability()) != (7, 2):
        return False
    if getattr(torchvision.ops.nms, "_ball_cpu_fallback", False):
        return True

    original_nms = torchvision.ops.nms

    def cpu_nms(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            indices = original_nms(
                boxes.detach().cpu(),
                scores.detach().cpu(),
                iou_threshold,
            )
            return indices.to(boxes.device)
        return original_nms(boxes, scores, iou_threshold)

    cpu_nms._ball_cpu_fallback = True
    torchvision.ops.nms = cpu_nms
    return True
