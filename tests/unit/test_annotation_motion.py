import numpy as np
import pytest
from src.orchestrator.annotation_motion import (
    box_iou,
    estimate_cumulative_motion,
    map_box,
    Overlay,
    _visible_fraction,
    overlays_from_front_rear_tracks,
    overlays_from_stitching_tracks,
)


def test_box_iou():
    b1 = (0.0, 0.0, 100.0, 100.0)
    b2 = (50.0, 0.0, 150.0, 100.0)
    # intersection 50x100 = 5000, union 10000 + 10000 - 5000 = 15000 -> 1/3
    assert box_iou(b1, b2) == pytest.approx(1.0 / 3.0)

    # Disjoint
    b3 = (200.0, 200.0, 300.0, 300.0)
    assert box_iou(b1, b3) == pytest.approx(0.0)


def test_map_box_identity():
    C = np.array([np.eye(3), np.eye(3)])
    box = (10.0, 20.0, 100.0, 150.0)
    mapped = map_box(box, C, 0, 1)
    assert mapped == pytest.approx(box)


def test_map_box_translation():
    # C[0] is identity, C[1] has translation +10 in x, +5 in y
    M = np.eye(3)
    M[0, 2] = 10.0
    M[1, 2] = 5.0
    C = np.array([np.eye(3), M])
    box = (10.0, 20.0, 100.0, 150.0)
    mapped = map_box(box, C, 0, 1)
    assert mapped == pytest.approx((20.0, 25.0, 110.0, 155.0))


def test_visible_fraction():
    # Fully visible
    box_inside = (10.0, 10.0, 110.0, 110.0)
    assert _visible_fraction(box_inside, 200, 200) == pytest.approx(1.0)

    # Half out on right
    box_half = (150.0, 0.0, 250.0, 100.0)
    assert _visible_fraction(box_half, 200, 200) == pytest.approx(0.5)

    # Completely outside
    box_outside = (250.0, 250.0, 300.0, 300.0)
    assert _visible_fraction(box_outside, 200, 200) == pytest.approx(0.0)


def test_overlay_box_at_interpolation():
    C = np.array([np.eye(3) for _ in range(20)])
    # 2 anchors: frame 0 at (300, 100, 500, 300), frame 10 at (320, 100, 520, 300)
    # (Inside w0=1000, so center_x ~ 0.40*w0, within 0.10 to 0.78 inspection area)
    anchors = {0: (300.0, 100.0, 500.0, 300.0), 10: (320.0, 100.0, 520.0, 300.0)}
    ov = Overlay(label="TEST1234567", is_valid=True, anchors=anchors, first_frame=0, last_frame=10)

    # At frame 5, should be exactly halfway: x1=310, x2=510
    b5 = ov.box_at(5, C, w0=1000, h0=800)
    assert b5 is not None
    assert b5 == pytest.approx((310.0, 100.0, 510.0, 300.0))


def test_golden_anchor_outlier_rejection():
    # C is identity across 50 frames
    C = np.array([np.eye(3) for _ in range(50)])

    class DummyBBox:
        def __init__(self, x1, y1, x2, y2):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    class DummyTrack:
        def __init__(self):
            self.clean_text = "MSMU6035818"
            self.iso_text = "45G1"
            self.is_valid = True
            self.track_id = 1
            self.best_frame_idx = 10  # 1-based (0-based = 9)
            self.best_nb = DummyBBox(500, 200, 600, 400)
            self.best_cb = DummyBBox(400, 100, 800, 800)
            # Anchor 1: frame 10 (f0=9) is at 500 (golden)
            # Anchor 2: frame 15 (f0=14) is at 505 (valid container tracking)
            # Anchor 3: frame 30 (f0=29) jumped to 900 (neighbor container!)
            self.num_history = {
                10: (500, 200, 600, 400),
                15: (505, 200, 605, 400),
                30: (900, 200, 1000, 400),
            }
            self.history = {10: None, 15: None, 30: None}

    tracks = [DummyTrack()]
    overlays = overlays_from_front_rear_tracks(tracks, lambda t: t.clean_text, C=C)
    assert len(overlays) == 1
    # Frame 29 (anchor 3 at 900) must be REJECTED because it's ~400px away from golden 500!
    assert 29 not in overlays[0].anchors
    assert 9 in overlays[0].anchors
    assert 14 in overlays[0].anchors


def test_container_centric_pinned_tracking():
    # Test that badge stays locked to container body without drifting or jumping
    C = np.array([np.eye(3) for _ in range(50)])
    # Container moves from x1=1000 at frame 10 to x1=1500 at frame 20 (w=800, h=1000)
    cont_hist = {
        10: (1000.0, 500.0, 1800.0, 1500.0),
        20: (1500.0, 500.0, 2300.0, 1500.0),
    }
    # Badge relative: right side of container, rx1=0.80, ry1=0.20, rx2=0.95, ry2=0.70
    rel_badge = (0.80, 0.20, 0.95, 0.70)
    ov = Overlay(
        label="UETU6596922",
        is_valid=True,
        anchors={},
        first_frame=10,
        last_frame=20,
        container_history=cont_hist,
        rel_badge=rel_badge,
    )

    # At frame 10: badge should be exactly at 1000 + 0.80*800 = 1640
    cb10 = ov.container_box_at(10, C, w0=3840, h0=2160)
    assert cb10 == pytest.approx((1000.0, 500.0, 1800.0, 1500.0))
    b10 = ov.box_at(10, C, w0=3840, h0=2160)
    assert b10 is not None
    assert b10[0] == pytest.approx(1640.0)
    assert b10[2] == pytest.approx(1760.0)
    assert b10[1] == pytest.approx(700.0)
    assert b10[3] == pytest.approx(1200.0)

    # At frame 15: halfway interpolation: cont x1=1250, badge x1=1250 + 640 = 1890
    cb15 = ov.container_box_at(15, C, w0=3840, h0=2160)
    assert cb15 == pytest.approx((1250.0, 500.0, 2050.0, 1500.0))
    b15 = ov.box_at(15, C, w0=3840, h0=2160)
    assert b15 is not None
    assert b15[0] == pytest.approx(1890.0)
    assert b15[2] == pytest.approx(2010.0)

    # At frame 20: badge x1=1500 + 640 = 2140
    cb20 = ov.container_box_at(20, C, w0=3840, h0=2160)
    assert cb20 == pytest.approx((1500.0, 500.0, 2300.0, 1500.0))
    b20 = ov.box_at(20, C, w0=3840, h0=2160)
    assert b20 is not None
    assert b20[0] == pytest.approx(2140.0)
    assert b20[2] == pytest.approx(2260.0)


def test_front_rear_edge_clipping_rigid_width():
    """Edge-clipped YOLO observations must NOT collapse container width or slide num box sideways."""
    class DummyBox:
        def __init__(self, x1, y1, x2, y2):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
            self.width = x2 - x1
            self.height = y2 - y1

    class EdgeTrack:
        def __init__(self):
            self.clean_text = "TCLU7737929"
            self.iso_text = "45G1"
            self.is_valid = True
            self.track_id = 1
            self.best_frame_idx = 31  # 0-based = 30
            # Golden full interior width = 900 (1500..2400), number at right side 2100..2260
            self.best_cb = DummyBox(1500, 600, 2400, 1600)
            self.best_nb = DummyBox(2100, 700, 2260, 1200)
            self.history = {
                11: DummyBox(28, 600, 450, 1600),     # 0-based 10: half off left edge with x1=28 (> 25)
                31: DummyBox(1500, 600, 2400, 1600),  # 0-based 30: golden full interior
                51: DummyBox(3300, 600, 3840, 1600),  # 0-based 50: half off right edge (clipped width 540)
            }
            self.num_history = {31: (2100, 700, 2260, 1200)}

    C = np.array([np.eye(3) for _ in range(60)])
    ov = overlays_from_front_rear_tracks([EdgeTrack()], lambda t: t.clean_text, C=C)[0]
    # Reconstructed left-clipped container box at frame 10 keeps physical right edge 450 and width 900 -> x1=-450
    c10 = ov.container_history[10]
    assert c10[0] == pytest.approx(-450.0)
    assert c10[2] - c10[0] == pytest.approx(900.0)
    # Only unclipped frame 30 stays in interior container_anchors
    assert 10 not in ov.container_anchors
    assert 30 in ov.container_anchors


def test_stitching_h_pano_to_vid_multi_anchor():
    """Stitching overlays must project container_number via H_pano_to_vid across entire trajectory."""
    H_f50 = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 100.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_f150 = np.array([[1.0, 0.0, 2200.0], [0.0, 1.0, 100.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    track_dict = {
        "route_order": 1,
        "track_id": 1,
        "clean_text": "TCLU7737929",
        "iso_text": "45G1",
        "is_valid": True,
        "pass_frame": 100,
        "panorama_container_bbox": [0.0, 0.0, 1200.0, 400.0],
        "detections": {
            "container_number": {"bbox": {"x": 900, "y": 80, "width": 80, "height": 200}, "confidence": 0.9}
        },
        "frame_registrations": [
            {"frame": 50, "H_pano_to_vid": H_f50.tolist(), "det_bbox": [200.0, 100.0, 1400.0, 500.0]},
            {"frame": 150, "H_pano_to_vid": H_f150.tolist(), "det_bbox": [2200.0, 100.0, 3400.0, 500.0]},
        ],
    }
    ov = overlays_from_stitching_tracks([track_dict])[0]
    assert ov.first_frame == 50
    assert ov.last_frame == 150
    assert 50 in ov.anchors and 150 in ov.anchors
    assert ov.anchors[50] == pytest.approx((1100.0, 180.0, 1180.0, 380.0))
    assert ov.anchors[150] == pytest.approx((3100.0, 180.0, 3180.0, 380.0))


def test_stitching_offscreen_proj_nb_with_clipped_det_bbox():
    """Off-screen H_pano_to_vid number box must NOT be dropped just because det_bbox is screen-clipped."""
    H_f10 = np.array([[1.0, 0.0, 1000.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    track_dict = {
        "route_order": 1,
        "track_id": 1,
        "clean_text": "MSMU6764912",
        "is_valid": True,
        "pass_frame": 50,
        "panorama_container_bbox": {"x": 0.0, "y": 0.0, "width": 4000.0, "height": 800.0},
        "detections": {
            "container_number": {"bbox": {"x": 3800, "y": 100, "width": 100, "height": 400}, "confidence": 0.9}
        },
        "frame_registrations": [
            # det_bbox is screen-clipped at 3840, while proj_cb reaches 5000 and proj_nb sits at 4800..4900!
            {"frame": 10, "H_pano_to_vid": H_f10.tolist(), "det_bbox": [1000.0, 0.0, 3840.0, 800.0]},
        ],
    }
    ov = overlays_from_stitching_tracks([track_dict])[0]
    assert 10 in ov.anchors
    assert ov.anchors[10][0] == pytest.approx(4800.0)
    assert ov.container_anchors[10][2] == pytest.approx(5000.0)


def test_roi_lk_tracks_synthetic_translation(tmp_path):
    """ROI-local FB-LK must propagate container and number boxes forward & backward smoothly."""
    import cv2
    from src.orchestrator.annotation_motion import render_overlay_video

    vid_path = tmp_path / "synth_motion.mp4"
    w, h, n_frames = 640, 360, 70
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    rng = np.random.default_rng(42)
    tex = rng.integers(30, 220, size=(180, 260, 3), dtype=np.uint8)
    # Add crisp stripes inside textured container
    for gx in range(10, 250, 20):
        cv2.line(tex, (gx, 5), (gx, 175), (255, 255, 255), 2)

    vx = 4.0
    x0 = 40.0
    for f in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx = int(round(x0 + vx * f))
        x_end = min(w, cx + 260)
        if x_end > cx and cx < w:
            src_w = x_end - max(0, cx)
            src_x0 = max(0, -cx)
            frame[90:270, max(0, cx):x_end] = tex[:, src_x0:src_x0 + src_w]
        writer.write(frame)
    writer.release()

    # Anchor ONLY at frame 35 -> must track cleanly backward to frame <= 20 and forward to frame >= 50
    cx35 = x0 + vx * 35
    ov = Overlay(
        label="SYNTH000001",
        is_valid=True,
        anchors={35: (cx35 + 180.0, 110.0, cx35 + 230.0, 210.0)},
        container_anchors={35: (cx35, 90.0, cx35 + 260.0, 270.0)},
        golden_frame=35,
        golden_container_box=(cx35, 90.0, cx35 + 260.0, 270.0),
        first_frame=35,
        last_frame=35,
    )
    out_mp4 = tmp_path / "synth_out.mp4"
    ok = render_overlay_video(str(vid_path), str(out_mp4), [ov])
    assert ok and out_mp4.is_file()
    for check_f in (20, 35, 50):
        assert check_f in ov._resolved_num
        expected_x1 = x0 + vx * check_f + 180.0
        assert abs(ov._resolved_num[check_f][0] - expected_x1) < 5.0


def test_stitching_mismatched_4k_anchor_and_edge_snap():
    """Do not let misaligned 4K ISO bbox replace clean Stage-4 number bbox; snap unclipped det_bbox edges."""
    H_f10 = np.eye(3, dtype=np.float64)
    track_dict = {
        "route_order": 1,
        "track_id": 1,
        "clean_text": "UETU7365661",
        "is_valid": True,
        "pass_frame": 10,
        # Bloated union panorama container bbox (+150px vertically)
        "panorama_container_bbox": [0.0, -50.0, 2000.0, 1150.0],
        "detections": {
            "container_number": {"bbox": {"x": 1600, "y": 200, "width": 70, "height": 380}, "confidence": 0.9},
            "iso_code": {"bbox": {"x": 1730, "y": 210, "width": 70, "height": 240}, "confidence": 0.9},
        },
        "direct_4k_anchor": {
            "frame": 10,
            # Shifted right onto adjacent ISO code (0 IoU with Stage-4 1600..1670)
            "num_box": [1740.0, 210.0, 1810.0, 460.0],
            # Screen-clipped det_bbox must not overwrite unclipped proj_cb
            "cont_box": [0.0, 100.0, 1750.0, 950.0],
        },
        "frame_registrations": [
            {"frame": 10, "H_pano_to_vid": H_f10.tolist(), "det_bbox": [0.0, 100.0, 1750.0, 950.0]},
        ],
    }
    ov = overlays_from_stitching_tracks([track_dict])[0]
    # Stage-4 number box 1600..1670 preserved instead of 1740..1810
    assert ov.anchors[10][0] == pytest.approx(1600.0)
    # Unclipped top/bottom/right borders snapped cleanly to det_bbox (100..950, right=1750), left kept at proj_cb (0.0)
    assert ov.container_anchors[10] == pytest.approx((0.0, 100.0, 1750.0, 950.0))


def test_edge_container_presence_without_center_in_view():
    """Green container box remains visible at screen edges even when unclipped center is off-screen."""
    ov = Overlay(
        label="CAAU6955064",
        is_valid=True,
        anchors={10: (400.0, 950.0, 500.0, 1550.0)},
        container_anchors={10: (-3000.0, 800.0, 800.0, 2000.0)},
        first_frame=10,
        last_frame=10,
        min_container_visible=0.10,
    )
    cb = ov.container_box_at(10, C=None, w0=3840, h0=2160)
    assert cb is not None
    assert cb[2] == pytest.approx(800.0)


def test_late_number_anchor_does_not_shrink_to_1px_on_clipped_container(tmp_path):
    """When cont starts earlier on a clipped fragment than num, rigid width is preserved without 1px collapse."""
    import cv2
    from src.orchestrator.annotation_motion import render_overlay_video

    vid_path = tmp_path / "clipped_entry.mp4"
    w, h, n_frames = 640, 360, 25
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    rng = np.random.default_rng(7)
    tex = rng.integers(40, 220, size=(180, 500, 3), dtype=np.uint8)
    for f in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cx = 100 + f * 4
        x1, x2 = max(0, cx), min(w, cx + 500)
        frame[80:260, x1:x2] = tex[:, (x1 - cx):(x2 - cx)]
        writer.write(frame)
    writer.release()

    # Container starts at f=5 (clipped width look), number starts at f=15 (rigid width 50px)
    ov = Overlay(
        label="R002TEST",
        is_valid=True,
        anchors={15: (220.0, 100.0, 270.0, 220.0)},
        container_anchors={
            5: (120.0, 80.0, 620.0, 260.0),
            15: (160.0, 80.0, 660.0, 260.0),
        },
        golden_frame=15,
        golden_container_box=(160.0, 80.0, 660.0, 260.0),
        first_frame=5,
        last_frame=15,
    )
    out_mp4 = tmp_path / "clipped_entry_out.mp4"
    assert render_overlay_video(str(vid_path), str(out_mp4), [ov])
    for fk in range(2, 16):
        b = ov._resolved_num.get(fk)
        assert b is not None
        assert (b[2] - b[0]) == pytest.approx(50.0, abs=2.0)


def test_multi_overlay_late_number_anchor_no_keyerror(tmp_path):
    """Multi-container video must look up each overlay's own num_src (regression for KeyError on frame index)."""
    import cv2
    from src.orchestrator.annotation_motion import render_overlay_video

    vid_path = tmp_path / "multi_late.mp4"
    w, h, n_frames = 320, 180, 26
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    rng = np.random.default_rng(11)
    tex = rng.integers(40, 220, size=(80, 120, 3), dtype=np.uint8)
    for f in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[40:120, 80:200] = tex
        writer.write(frame)
    writer.release()

    ov1 = Overlay(
        label="R001_LATE",
        is_valid=True,
        anchors={12: (100.0, 50.0, 130.0, 100.0)},
        container_anchors={3: (80.0, 40.0, 200.0, 120.0), 12: (80.0, 40.0, 200.0, 120.0)},
        golden_frame=12,
        golden_container_box=(80.0, 40.0, 200.0, 120.0),
        first_frame=3,
        last_frame=12,
    )
    ov2 = Overlay(
        label="R002_LATER",
        is_valid=True,
        anchors={20: (105.0, 52.0, 135.0, 102.0)},
        container_anchors={18: (82.0, 40.0, 202.0, 120.0), 20: (82.0, 40.0, 202.0, 120.0)},
        golden_frame=20,
        golden_container_box=(82.0, 40.0, 202.0, 120.0),
        first_frame=18,
        last_frame=20,
    )
    out_mp4 = tmp_path / "multi_late_out.mp4"
    assert render_overlay_video(str(vid_path), str(out_mp4), [ov1, ov2])
    assert ov1._resolved_num.get(3) is not None
    assert ov2._resolved_num.get(18) is not None





