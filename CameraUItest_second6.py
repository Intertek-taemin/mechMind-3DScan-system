import os
import sys
import numpy as np
from datetime import datetime

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *

try:
    import open3d as o3d
except ImportError:
    o3d = None

# pyqtgraph 3D 뷰어
try:
    import pyqtgraph as pg
    from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem
except ImportError:
    pg = None
    GLViewWidget = None
    GLScatterPlotItem = None

# 깊이 TIFF / 이미지 읽기용
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QSizePolicy, QButtonGroup, QMessageBox,
    QDialog, QListWidget, QLineEdit  # ★ 여기 추가
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QPixmap, QImage

# --------------------------------------------------------------------
# ★ 실제 카메라를 쓸 때는 True 로 바꾸기.
# --------------------------------------------------------------------
USE_REAL_CAMERA = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_PLY_PATH = os.path.join(BASE_DIR, "sample", "sample_point.ply")
SAMPLE_TIFF_PATH = os.path.join(BASE_DIR, "sample", "sample_depth.tiff")
SAMPLE_IMG_PATH = os.path.join(BASE_DIR, "sample", "sampleimg.png")

# 결과 저장 폴더
RESULT_DIR = os.path.join(BASE_DIR, "result")


# ----------------- 깊이값 → 레인보우 컬러맵 -----------------
def hsv_to_rgb_np(h, s, v):
    """h,s,v in [0,1] numpy 배열 → r,g,b in [0,1] numpy 배열."""
    h = np.mod(h, 1.0)
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    mask = (i_mod == 0)
    r[mask], g[mask], b[mask] = v[mask], t[mask], p[mask]

    mask = (i_mod == 1)
    r[mask], g[mask], b[mask] = q[mask], v[mask], p[mask]

    mask = (i_mod == 2)
    r[mask], g[mask], b[mask] = p[mask], v[mask], t[mask]

    mask = (i_mod == 3)
    r[mask], g[mask], b[mask] = p[mask], q[mask], v[mask]

    mask = (i_mod == 4)
    r[mask], g[mask], b[mask] = t[mask], p[mask], v[mask]

    mask = (i_mod == 5)
    r[mask], g[mask], b[mask] = v[mask], p[mask], q[mask]

    return r, g, b


def depth_to_color_image(depth_np: np.ndarray) -> np.ndarray:
    """
    depth_np: 2D float32 배열 (깊이값)
    return: HxWx3 uint8 RGB 컬러맵 이미지
    """
    depth = depth_np.astype(np.float32)

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    if depth.ndim == 3:
        depth = depth[..., 0]

    mask = np.isfinite(depth) & (depth > 0)
    if not np.any(mask):
        h, w = depth.shape
        return np.zeros((h, w, 3), np.uint8)

    d_min = float(depth[mask].min())
    d_max = float(depth[mask].max())

    if d_max <= d_min:
        h, w = depth.shape
        return np.zeros((h, w, 3), np.uint8)

    norm = (depth - d_min) / (d_max - d_min + 1e-6)
    norm = np.clip(norm, 0.0, 1.0)

    # 가까운 곳(값 작음) = 빨강, 먼 곳(값 큼) = 파랑
    h = (norm) * (2.0 / 3.0)
    s = np.ones_like(h)
    v = np.ones_like(h)

    r, g, b = hsv_to_rgb_np(h, s, v)
    rgb = np.stack([r, g, b], axis=-1)

    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    rgb_uint8[~mask] = 0
    return rgb_uint8


class CameraSelectDialog(QDialog):
    """카메라 목록을 보여주고, 하나 선택해서 OK하는 다이얼로그."""
    def __init__(self, camera_infos, parent=None):
        super().__init__(parent)

        self.setWindowTitle("카메라 연결")
        self.resize(400, 250)
        self.camera_infos = camera_infos
        self.selected_index = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        label = QLabel("확인된 카메라 목록...")
        main_layout.addWidget(label)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        main_layout.addWidget(self.list_widget)

        if not camera_infos:
            self.list_widget.addItem("연결된 카메라가 없습니다.")
            self.list_widget.setEnabled(False)
        else:
            for i, info in enumerate(camera_infos):
                try:
                    text = f"{i}: {info.modelName} | IP: {info.ipAddress} | SN: {info.serialNumber}"
                except Exception:
                    text = f"{i}: {info}"
                self.list_widget.addItem(text)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_connect = QPushButton("연결")
        self.btn_connect.setFixedWidth(100)
        self.btn_connect.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color:white; padding:6px; }"
        )
        self.btn_connect.clicked.connect(self.on_connect_clicked)

        if not camera_infos:
            self.btn_connect.setEnabled(False)

        btn_layout.addWidget(self.btn_connect)
        main_layout.addLayout(btn_layout)

        self.list_widget.itemDoubleClicked.connect(self.on_connect_clicked)

    def on_connect_clicked(self, *args):
        row = self.list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "선택 필요", "연결할 카메라를 선택하세요.")
            return
        self.selected_index = row
        self.accept()


class ScanViewer(QWidget):
    """
    3D / 2D 스캔 데이터 출력 영역.
    - Image 모드: 2D 컬러(또는 흑백) 이미지
    - Point 모드: pyqtgraph GLViewWidget으로 3D 포인트 클라우드
    - Depths 모드: 뎁스 컬러맵(QPixmap)
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

        self.label = QLabel("스캔 데이터 출력")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 16px;")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._layout.addWidget(self.label)

        self.gl_view = None
        self.gl_scatter = None

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def ensure_gl_view(self):
        """GLViewWidget이 없다면 생성하고, 라벨 대신 보이게 한다."""
        if GLViewWidget is None or GLScatterPlotItem is None:
            return False

        if self.gl_view is None:
            self.gl_view = GLViewWidget()
            self.gl_view.setBackgroundColor('k')
            self.gl_view.opts['fov'] = 60
            self.gl_view.opts['elevation'] = 25
            self.gl_view.opts['azimuth'] = 45
            self.gl_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._layout.addWidget(self.gl_view)

        self.label.hide()
        self.gl_view.show()
        return True

    def show_pointcloud(self, points: np.ndarray, colors: np.ndarray | None = None):
        """3D 포인트 클라우드를 pyqtgraph GLViewWidget에 그리기."""
        if not self.ensure_gl_view():
            self.label.setText("pyqtgraph / PyOpenGL 설치 필요: pip install pyqtgraph PyOpenGL")
            self.label.show()
            if self.gl_view is not None:
                self.gl_view.hide()
            return

        if points is None or points.size == 0:
            self.label.setText("표시할 포인트가 없습니다.")
            self.label.show()
            if self.gl_view is not None:
                self.gl_view.hide()
            return

        pts = points.astype(np.float32)

        max_points = 150_000
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
            if colors is not None:
                colors = colors[idx]

        center = pts.mean(axis=0)
        pts_centered = pts - center

        if colors is not None:
            cols = colors.astype(np.float32)
            if cols.max() > 1.0:
                cols = cols / 255.0
        else:
            cols = np.ones((pts.shape[0], 3), dtype=np.float32)

        if cols.ndim == 2 and cols.shape[1] == 3:
            alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
            cols_rgba = np.concatenate([cols, alpha], axis=1)
        else:
            cols_rgba = cols

        radius = float(np.linalg.norm(pts_centered, axis=1).max())
        if radius > 0:
            self.gl_view.opts['distance'] = radius * 2.5

        point_size = 2.0

        if self.gl_scatter is None:
            self.gl_scatter = GLScatterPlotItem(
                pos=pts_centered,
                color=cols_rgba,
                size=point_size,
                pxMode=True,
                glOptions='opaque',
            )
            self.gl_view.addItem(self.gl_scatter)
        else:
            self.gl_scatter.setData(
                pos=pts_centered,
                color=cols_rgba,
                size=point_size,
                pxMode=True,
            )
            self.gl_scatter.setGLOptions('opaque')

        if pg is not None:
            self.gl_view.opts['center'] = pg.Vector(0, 0, 0)

        self.gl_view.show()

    def show_image(self, pixmap: QPixmap):
        """2D 이미지 / 뎁스 컬러맵 표시."""
        if self.gl_view is not None:
            self.gl_view.hide()
        self.label.setPixmap(pixmap)
        self.label.setText("")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("3D 카메라 스캔 인터페이스 (Prototype)")
        self.resize(1100, 650)

        # -------------------------------------------------
        # 카메라 & 데이터 상태
        # -------------------------------------------------
        self.camera = None
        self.camera_connected = False

        # 최근 스캔 데이터 (우측)
        self.last_pointcloud = None
        self.last_colors = None
        self.last_depth_qimage: QImage | None = None
        self.last_depth_array: np.ndarray | None = None
        # ★ NEW: 최근 2D 컬러/그레이 이미지
        self.last_color_qimage: QImage | None = None

        # 비교 기준 데이터 (좌측)
        self.compare_pointcloud = None
        self.compare_colors = None
        self.compare_depth_qimage: QImage | None = None
        self.compare_depth_array: np.ndarray | None = None
        # ★ NEW: 비교용 2D 이미지
        self.compare_color_qimage: QImage | None = None

        self.show_diff_overlay = False

        # 마지막 촬영 시각 기록
        self.last_capture_dt: datetime | None = None
        self.last_capture_name: str | None = None  # 폴더/파일 이름용(YYYYMMDD_HHMMSS)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        central.setLayout(main_layout)

        # =======================
        # 상단: Point / Depths / Image + Save + 시간 + Exit
        # =======================
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        btn_style = """
            QPushButton {
                background-color: #3f4c55;
                color: white;
                padding: 6px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #0e5a7a;
            }
        """

        # ★ 버튼 순서/매핑: Point, Depths, Image
        self.btn_point = QPushButton("Point")
        self.btn_depth = QPushButton("Depths")
        self.btn_image = QPushButton("Image")

        for b in (self.btn_point, self.btn_depth, self.btn_image):
            b.setCheckable(True)
            b.setStyleSheet(btn_style)
            b.clicked.connect(self.on_mode_changed)

        self.btn_point.setChecked(True)  # 기본은 Point 모드

        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.btn_point)
        group.addButton(self.btn_depth)
        group.addButton(self.btn_image)

        top_layout.addWidget(self.btn_point)
        top_layout.addWidget(self.btn_depth)
        top_layout.addWidget(self.btn_image)

        # Save 버튼 & 촬영 시간 표시 라벨
        self.btn_save_files = QPushButton("Save")
        self.btn_save_files.setStyleSheet(btn_style)
        self.btn_save_files.setEnabled(False)  # 촬영 전에는 비활성화
        self.btn_save_files.clicked.connect(self.on_save_files_clicked)
        top_layout.addWidget(self.btn_save_files)

        self.label_last_capture = QLabel("이미지 미촬영")
        self.label_last_capture.setStyleSheet(
            "color:white; padding-left:8px; font-size:12px;"
        )
        self.label_last_capture.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        top_layout.addWidget(self.label_last_capture)

        top_layout.addStretch(1)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setStyleSheet(
            "QPushButton { background-color: red; color: white; "
            "font-weight: bold; padding: 6px; }"
        )
        self.btn_exit.setFixedWidth(80)
        self.btn_exit.clicked.connect(self.close)
        top_layout.addWidget(self.btn_exit)

        main_layout.addLayout(top_layout)

        # =======================
        # 중간: 좌/우 뷰 + 허용 오차 패널
        # =======================
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(10)

        # --- 좌: 비교 대상 뷰 ---
        left_col = QVBoxLayout()
        left_col.setSpacing(4)

        lbl_left_title = QLabel("저장된 비교 대상")
        lbl_left_title.setAlignment(Qt.AlignCenter)
        lbl_left_title.setStyleSheet(
            "background-color:#555555; color:white; padding:4px; font-size:13px;"
        )
        left_col.addWidget(lbl_left_title)

        self.compare_viewer = ScanViewer()
        self.compare_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_col.addWidget(self.compare_viewer)

        mid_layout.addLayout(left_col, 5)

        # --- 가운데: 최근 촬영 뷰 ---
        center_col = QVBoxLayout()
        center_col.setSpacing(4)

        lbl_center_title = QLabel("가장 최근에 촬영된 이미지")
        lbl_center_title.setAlignment(Qt.AlignCenter)
        lbl_center_title.setStyleSheet(
            "background-color:#555555; color:white; padding:4px; font-size:13px;"
        )
        center_col.addWidget(lbl_center_title)

        self.scan_viewer = ScanViewer()
        self.scan_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_col.addWidget(self.scan_viewer)

        mid_layout.addLayout(center_col, 5)

        # --- 우측: 허용 오차 & 로그 초기화 ---
        right_side = QVBoxLayout()
        right_side.setSpacing(10)

        lbl_tol_title = QLabel("허용 오차")
        lbl_tol_title.setAlignment(Qt.AlignCenter)
        lbl_tol_title.setStyleSheet(
            "background-color:#3f4c55; color:white; padding:4px; font-size:14px;"
        )
        right_side.addWidget(lbl_tol_title)

        self.label_tol_indicator = QLabel("-")
        self.label_tol_indicator.setAlignment(Qt.AlignCenter)
        self.label_tol_indicator.setStyleSheet(
            "background-color:white; color:black; font-size:32px; font-weight:bold;"
        )
        self.label_tol_indicator.setMinimumSize(70, 70)
        right_side.addWidget(self.label_tol_indicator)

        self.btn_diff_toggle = QPushButton("Diff 표시 OFF")
        self.btn_diff_toggle.setCheckable(True)
        self.btn_diff_toggle.setStyleSheet(
            "QPushButton { background-color:#777777; color:white; padding:4px; font-size:12px; }"
            "QPushButton:checked { background-color:#0e5a7a; }"
        )
        self.btn_diff_toggle.setFixedWidth(100)
        self.btn_diff_toggle.setFixedHeight(32)
        self.btn_diff_toggle.clicked.connect(self.on_diff_toggle_clicked)
        right_side.addWidget(self.btn_diff_toggle)

        # ===== 허용 오차 입력칸 (기본값 포함) =====
        lbl_dist = QLabel("오차 거리(mm)")
        lbl_dist.setAlignment(Qt.AlignCenter)
        lbl_dist.setStyleSheet("color:white; font-size:12px;")
        right_side.addWidget(lbl_dist)

        self.edit_tol_distance = QLineEdit("10")     # ★ 기본값 10mm
        self.edit_tol_distance.setAlignment(Qt.AlignCenter)
        self.edit_tol_distance.setMaximumWidth(100)
        right_side.addWidget(self.edit_tol_distance)

        lbl_ratio = QLabel("오차 비율(%)")
        lbl_ratio.setAlignment(Qt.AlignCenter)
        lbl_ratio.setStyleSheet("color:white; font-size:12px;")
        right_side.addWidget(lbl_ratio)

        self.edit_tol_ratio = QLineEdit("2")         # ★ 기본값 2%
        self.edit_tol_ratio.setAlignment(Qt.AlignCenter)
        self.edit_tol_ratio.setMaximumWidth(100)
        right_side.addWidget(self.edit_tol_ratio)

        # ===== 기준 적용 버튼 =====
        self.btn_apply_tol = QPushButton("Apply")
        self.btn_apply_tol.setStyleSheet(
            "QPushButton { background-color:#555555; color:white; "
            "padding:4px; font-size:12px; }"
        )
        self.btn_apply_tol.setFixedWidth(100)
        self.btn_apply_tol.setFixedHeight(28)
        self.btn_apply_tol.clicked.connect(self.on_apply_tolerance_clicked)
        right_side.addWidget(self.btn_apply_tol)

        right_side.addStretch(1)

        self.btn_refresh = QPushButton("로그 초기화")
        self.btn_refresh.clicked.connect(self.on_refresh_clicked)
        self.btn_refresh.setStyleSheet(
            "QPushButton { background-color:white; color:black; padding:4px; font-size:12px; }"
        )
        self.btn_refresh.setFixedWidth(90)
        self.btn_refresh.setFixedHeight(36)
        right_side.addWidget(self.btn_refresh)

        mid_layout.addLayout(right_side, 1)

        main_layout.addLayout(mid_layout, 5)

        # =======================
        # 하단: 로그 + 버튼들
        # =======================
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(10)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: white; color: black")
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bottom_layout.addWidget(self.log_text, 5)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)

        bottom_btn_style = """
            QPushButton {
                background-color: #0e5a7a;
                color: white;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #4a6c7a;
            }
        """

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setStyleSheet(bottom_btn_style)
        self.btn_connect.clicked.connect(self.connect_camera)
        btn_col.addWidget(self.btn_connect)

        self.btn_scan = QPushButton("Scan")
        self.btn_scan.setStyleSheet(bottom_btn_style)
        self.btn_scan.clicked.connect(self.scan_pointcloud)
        self.btn_scan.setEnabled(False)
        btn_col.addWidget(self.btn_scan)

        self.btn_save_compare = QPushButton("Save\nCompare\nData")
        self.btn_save_compare.setStyleSheet(bottom_btn_style)
        self.btn_save_compare.clicked.connect(self.on_save_compare_clicked)
        btn_col.addWidget(self.btn_save_compare)

        bottom_layout.addLayout(btn_col, 1)

        main_layout.addLayout(bottom_layout, 1)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(70, 70, 70))
        self.setPalette(pal)

        if pg is None or GLViewWidget is None:
            self.append_log("[ERROR] pyqtgraph / PyOpenGL 미설치. 3D 뷰어를 사용하려면 pip install pyqtgraph PyOpenGL")

    # ===== 로그 도우미 =====
    def append_log(self, text: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{now}] {text}")

    # ===== 모드 전환 =====
    def on_mode_changed(self):
        if self.btn_point.isChecked():
            # 좌측: 기준 포인트는 항상 일반 색으로 표시
            if self.compare_pointcloud is not None:
                self.update_compare_pointcloud_view()

            # 우측: Diff ON이면 차이 색상 뷰, 아니면 일반 뷰
            if self.last_pointcloud is not None:
                if self.show_diff_overlay and self.compare_pointcloud is not None:
                    self.update_recent_pointcloud_diff_view()
                else:
                    self.update_recent_pointcloud_view()

        elif self.btn_depth.isChecked():
            if self.compare_depth_qimage is not None:
                self.update_compare_depth_view()
            if self.last_depth_qimage is not None:
                self.update_recent_depth_view()
        elif self.btn_image.isChecked():
            if self.compare_color_qimage is not None:
                self.update_compare_image_view()
            if self.last_color_qimage is not None:
                self.update_recent_image_view()

    # ===== 로그 초기화 =====
    def on_refresh_clicked(self):
        self.log_text.clear()

    # ===== 카메라 연결 =====
    def connect_camera(self):
        if not USE_REAL_CAMERA:
            self.append_log("[INFO] (더미) 카메라 연결을 생략합니다. 샘플 파일 모드로 동작합니다.")
            self.camera_connected = True
            self.btn_scan.setEnabled(True)
            return

        try:
            camera_infos = Camera.discover_cameras()

            dlg = CameraSelectDialog(camera_infos, self)
            result = dlg.exec()

            if not camera_infos:
                self.append_log("[ERROR] 검색된 카메라가 없습니다.")
                return

            if result != QDialog.Accepted or dlg.selected_index is None:
                return

            idx = dlg.selected_index

            self.camera = Camera()
            error_status = self.camera.connect(camera_infos[idx])

            if not error_status.is_ok():
                try:
                    show_error(error_status)
                except Exception:
                    pass
                self.append_log(f"[ERROR] 카메라 연결 실패: {error_status}")
                self.camera_connected = False
                self.btn_scan.setEnabled(False)
                return

            self.camera_connected = True
            self.btn_scan.setEnabled(True)

        except Exception as e:
            self.append_log(f"[ERROR] 카메라 연결 중 예외: {e}")
            self.camera_connected = False
            self.btn_scan.setEnabled(False)

    # ===== 스캔 =====
    def scan_pointcloud(self):
        if USE_REAL_CAMERA and not self.camera_connected:
            self.append_log("[ERROR] 카메라가 연결되지 않았습니다.")
            return

        self.append_log("촬영 실행: 2D + Depth + Point 데이터를 스캔합니다.")
        try:
            self.capture_both_depth_and_point()
        except Exception as e:
            self.append_log(f"[ERROR] 캡쳐 중 오류: {e}")
            return

        now_dt = datetime.now()
        self.last_capture_dt = now_dt
        self.last_capture_name = now_dt.strftime("%Y%m%d_%H%M%S")
        self.label_last_capture.setText(now_dt.strftime("%Y-%m-%d %H:%M:%S"))
        self.btn_save_files.setEnabled(True)

        if self.btn_point.isChecked():
            if self.last_pointcloud is None:
                self.append_log("[ERROR] 포인트 데이터가 없습니다.")
                return

            # Diff ON & 비교 데이터가 있으면 diff 뷰 유지
            if self.show_diff_overlay and self.compare_pointcloud is not None:
                self.update_recent_pointcloud_diff_view()
            else:
                self.update_recent_pointcloud_view()

        elif self.btn_depth.isChecked():
            if self.last_depth_qimage is None:
                self.append_log("[ERROR] 뎁스 데이터가 없습니다.")
                return
            self.update_recent_depth_view()
        elif self.btn_image.isChecked():
            if self.last_color_qimage is None:
                self.append_log("[ERROR] 2D 이미지가 없습니다.")
                return
            self.update_recent_image_view()

        self.update_tolerance_display()

    # ===== 스캔 데이터 파일로 저장 =====
    def on_save_files_clicked(self):
        """
        result/날짜시간/ 폴더에
        - PNG : 2D 카메라 이미지 (gradient 없는 일반 사진)
        - TIFF: 원본 뎁스 배열
        - PLY : 포인트 클라우드
        를 저장.
        """
        if self.last_capture_dt is None or self.last_capture_name is None:
            self.append_log("[ERROR] 저장할 스캔이 없습니다. 먼저 Scan을 실행하세요.")
            return

        if (
            self.last_pointcloud is None
            and self.last_depth_array is None
            and self.last_color_qimage is None
        ):
            self.append_log("[ERROR] 저장할 데이터가 없습니다.")
            return

        os.makedirs(RESULT_DIR, exist_ok=True)
        save_dir = os.path.join(RESULT_DIR, self.last_capture_name)
        os.makedirs(save_dir, exist_ok=True)

        base = self.last_capture_name

        # ★ PNG: 2D 카메라 이미지 (sampleimg 같은 것) -----------------
        if self.last_color_qimage is not None:
            png_path = os.path.join(save_dir, f"{base}.png")
            ok = self.last_color_qimage.save(png_path, "PNG")
            if ok:
                self.append_log(f"PNG(2D 이미지) 저장 완료: {png_path}")
            else:
                self.append_log("[ERROR] PNG 저장 실패")
        else:
            self.append_log("[WARN] PNG 저장 생략: 2D 이미지 없음")

        # (선택) 뎁스 컬러맵 PNG도 보고 싶으면 아래 주석을 풀면 됨
        # if self.last_depth_qimage is not None:
        #     depth_png_path = os.path.join(save_dir, f"{base}_depth.png")
        #     if self.last_depth_qimage.save(depth_png_path, "PNG"):
        #         self.append_log(f"뎁스 컬러맵 PNG 저장: {depth_png_path}")

        # TIFF: 원본 뎁스 배열 ------------------------------------------
        if self.last_depth_array is not None:
            tiff_path = os.path.join(save_dir, f"{base}.tiff")
            try:
                saved = False
                depth = self.last_depth_array.astype(np.float32)
                if Image is not None:
                    img = Image.fromarray(depth)
                    img.save(tiff_path, format="TIFF")
                    saved = True
                elif cv2 is not None:
                    cv2.imwrite(tiff_path, depth)
                    saved = True

                if saved:
                    self.append_log(f"TIFF 저장 완료: {tiff_path}")
                else:
                    self.append_log("[ERROR] TIFF 저장 실패: PIL / OpenCV 미설치")
            except Exception as e:
                self.append_log(f"[ERROR] TIFF 저장 중 예외: {e}")
        else:
            self.append_log("[WARN] TIFF 저장 생략: 뎁스 배열 없음")

        # PLY: 포인트 클라우드 ------------------------------------------
        if self.last_pointcloud is not None and self.last_pointcloud.size > 0:
            ply_path = os.path.join(save_dir, f"{base}.ply")
            try:
                pts = self.last_pointcloud
                cols = self.last_colors

                if o3d is not None:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    if cols is not None:
                        cols_to_use = cols.astype(np.float32)
                        if cols_to_use.max() > 1.0:
                            cols_to_use = cols_to_use / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(cols_to_use)
                    o3d.io.write_point_cloud(ply_path, pcd)
                else:
                    # open3d가 없으면 간단한 ASCII PLY로 직접 저장
                    n = pts.shape[0]
                    has_color = cols is not None and cols.shape[0] == n

                    with open(ply_path, "w", encoding="utf-8") as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {n}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        if has_color:
                            f.write("property uchar red\n")
                            f.write("property uchar green\n")
                            f.write("property uchar blue\n")
                        f.write("end_header\n")

                        for i in range(n):
                            x, y, z = pts[i]
                            if has_color:
                                c = cols[i].astype(np.float32)
                                if c.max() <= 1.0:
                                    c = c * 255.0
                                r, g, b = [
                                    int(np.clip(v, 0, 255)) for v in c
                                ]
                                f.write(f"{x} {y} {z} {r} {g} {b}\n")
                            else:
                                f.write(f"{x} {y} {z}\n")

                self.append_log(f"PLY 저장 완료: {ply_path}")
            except Exception as e:
                self.append_log(f"[ERROR] PLY 저장 중 예외: {e}")
        else:
            self.append_log("[WARN] PLY 저장 생략: 포인트 클라우드 없음")

        self.append_log(f"스캔 데이터 저장 완료: {save_dir}")

    # ===== 비교 데이터 저장 =====
    def on_save_compare_clicked(self):
        if (
            self.last_pointcloud is None
            and self.last_depth_array is None
            and self.last_color_qimage is None
        ):
            self.append_log("[ERROR] 비교 데이터로 저장할 스캔이 없습니다.")
            return

        self.compare_pointcloud = None if self.last_pointcloud is None else self.last_pointcloud.copy()
        self.compare_colors = None if self.last_colors is None else self.last_colors.copy()

        self.compare_depth_array = None if self.last_depth_array is None else self.last_depth_array.copy()
        self.compare_depth_qimage = None if self.last_depth_qimage is None else self.last_depth_qimage.copy()

        # ★ 2D 이미지 비교용도 같이 저장
        self.compare_color_qimage = None if self.last_color_qimage is None else self.last_color_qimage.copy()

        if self.btn_point.isChecked():
            self.update_compare_pointcloud_view()
        elif self.btn_depth.isChecked():
            self.update_compare_depth_view()
        elif self.btn_image.isChecked():
            self.update_compare_image_view()

        self.append_log("비교 데이터 저장 완료.")
        self.update_tolerance_display()

    # ===== 실제 캡쳐 동작 =====
    def capture_both_depth_and_point(self):
        """
        2D 이미지 + Depth map + Point cloud 를 한 번에 받아서
        '최근 데이터' 변수들(last_*)에 저장.
        """
        if USE_REAL_CAMERA:
            frame2d_and_3d = Frame2DAnd3D()
            show_error(self.camera.capture_2d_and_3d(frame2d_and_3d))

            # --- 2D 컬러 이미지 ---
            try:
                frame_2d = frame2d_and_3d.frame_2d()
                color_img = frame_2d.get_color_image()
                color_np = color_img.data().copy()

                if color_np.ndim == 3:
                    # BGR 또는 BGRA → RGB
                    if color_np.shape[2] == 4:
                        bgr = color_np[..., :3]
                    else:
                        bgr = color_np
                    rgb = bgr[..., ::-1].copy()
                    h, w, _ = rgb.shape
                    qimg_color = QImage(
                        rgb.data, w, h, 3 * w, QImage.Format_RGB888
                    ).copy()
                    self.last_color_qimage = qimg_color
                else:
                    self.last_color_qimage = None
            except Exception as e:
                self.last_color_qimage = None
                self.append_log(f"[ERROR] 2D 컬러 이미지 변환 실패: {e}")

            # --- Depth & Point ---
            frame_3d = frame2d_and_3d.frame_3d()

            depth_map = frame_3d.get_depth_map()
            depth_np = depth_map.data().copy().astype(np.float32)
            self.last_depth_array = depth_np

            rgb_uint8 = depth_to_color_image(depth_np)

            mask_valid = np.isfinite(depth_np) & (depth_np > 0)
            if np.any(mask_valid):
                ys, xs = np.where(mask_valid)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1

                pad_y = int(0.05 * depth_np.shape[0])
                pad_x = int(0.05 * depth_np.shape[1])

                y0 = max(0, y0 - pad_y)
                y1 = min(depth_np.shape[0], y1 + pad_y)
                x0 = max(0, x0 - pad_x)
                x1 = min(depth_np.shape[1], x1 + pad_x)

                rgb_uint8 = rgb_uint8[y0:y1, x0:x1]

            rgb_uint8 = np.ascontiguousarray(rgb_uint8)
            h, w, _ = rgb_uint8.shape
            qimg = QImage(rgb_uint8.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.last_depth_qimage = qimg

            pc = frame_3d.get_untextured_point_cloud()
            pc_np = pc.data().copy()

            pts = pc_np
            if pts.ndim == 3:
                pts = pts.reshape(-1, 3)

            mask = np.isfinite(pts).all(axis=1)
            mask &= (pts[:, 2] > 0)
            pts = pts[mask]

            self.last_pointcloud = pts
            self.last_colors = None

        else:
            # === 샘플 파일 모드 ===

            # 3D 포인트 클라우드
            if o3d is not None:
                pcd_raw = o3d.io.read_point_cloud(SAMPLE_PLY_PATH)
                pts_np = np.asarray(pcd_raw.points)
                mask = np.isfinite(pts_np).all(axis=1)
                pts = pts_np[mask]
                self.last_pointcloud = pts

                if pcd_raw.has_colors():
                    cols_np = np.asarray(pcd_raw.colors)
                    self.last_colors = cols_np[mask]
                else:
                    self.last_colors = None
            else:
                self.last_pointcloud = None
                self.last_colors = None

            # Depth TIFF
            depth_np = None
            if cv2 is not None:
                d = cv2.imread(SAMPLE_TIFF_PATH, cv2.IMREAD_UNCHANGED)
                if d is not None:
                    depth_np = d.astype(np.float32)
            if depth_np is None and Image is not None:
                try:
                    img = Image.open(SAMPLE_TIFF_PATH)
                    depth_np = np.array(img).astype(np.float32)
                except Exception:
                    depth_np = None

            if depth_np is not None:
                self.last_depth_array = depth_np
                rgb_uint8 = depth_to_color_image(depth_np)
                h, w, _ = rgb_uint8.shape
                qimg = QImage(
                    rgb_uint8.data, w, h, 3 * w,
                    QImage.Format_RGB888
                ).copy()
                self.last_depth_qimage = qimg
            else:
                self.last_depth_array = None
                self.last_depth_qimage = None

            # 샘플 2D 이미지 (sampleimg.png 그대로 사용)
            self.last_color_qimage = None
            if Image is not None:
                try:
                    img = Image.open(SAMPLE_IMG_PATH).convert("RGB")
                    arr = np.array(img)
                    h, w, _ = arr.shape
                    qimg_color = QImage(
                        arr.data, w, h, 3 * w, QImage.Format_RGB888
                    ).copy()
                    self.last_color_qimage = qimg_color
                except Exception:
                    self.last_color_qimage = None

            if self.last_color_qimage is None and cv2 is not None:
                try:
                    img = cv2.imread(SAMPLE_IMG_PATH, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, _ = img.shape
                        qimg_color = QImage(
                            img.data, w, h, 3 * w, QImage.Format_RGB888
                        ).copy()
                        self.last_color_qimage = qimg_color
                except Exception:
                    self.last_color_qimage = None

            if self.last_color_qimage is None:
                self.append_log("[WARN] 샘플 2D 이미지 로드 실패: sampleimg.png 확인 필요")

    # ===== 포인트 클라우드 뷰 갱신 =====
    def _update_viewer_with_pointcloud(self, viewer: ScanViewer,
                                       pts: np.ndarray | None,
                                       cols: np.ndarray | None):
        if pts is None or pts.size == 0:
            viewer.label.setText("유효 포인트 없음")
            viewer.label.show()
            return

        pts_vis = pts
        cols_vis = cols

        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if cols is not None:
                pcd.colors = o3d.utility.Vector3dVector(cols)

            center = pcd.get_center()
            pcd.translate(-center)
            base_R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            pcd.rotate(base_R, center=(0, 0, 0))

            pts_vis = np.asarray(pcd.points)
            if pcd.has_colors():
                cols_vis = np.asarray(pcd.colors)
            else:
                cols_vis = None

        if cols_vis is None and pts_vis is not None and pts_vis.size > 0:
            z = pts_vis[:, 2].astype(np.float32)
            mask = np.isfinite(z)
            if np.any(mask):
                z_valid = z[mask]
                z_min = float(z_valid.min())
                z_max = float(z_valid.max())
                denom = (z_max - z_min) + 1e-6

                norm = (z - z_min) / denom
                norm = np.clip(norm, 0.0, 1.0)

                h = (1 - norm) * (2.0 / 3.0)
                s = np.ones_like(h, dtype=np.float32)
                v = np.ones_like(h, dtype=np.float32)

                r, g, b = hsv_to_rgb_np(h, s, v)
                cols_vis = np.stack([r, g, b], axis=-1).astype(np.float32)
            else:
                cols_vis = None

        viewer.show_pointcloud(pts_vis, cols_vis)

    def update_recent_pointcloud_view(self):
        self._update_viewer_with_pointcloud(
            self.scan_viewer, self.last_pointcloud, self.last_colors
        )

    def update_compare_pointcloud_view(self):
        self._update_viewer_with_pointcloud(
            self.compare_viewer, self.compare_pointcloud, self.compare_colors
        )

    # ===== QImage 뷰 갱신 공통 =====
    def _update_viewer_with_qimage(self, viewer: ScanViewer,
                                   qimg: QImage | None,
                                   empty_text: str):
        if qimg is None:
            viewer.label.setText(empty_text)
            viewer.label.show()
            return

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            viewer.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        viewer.show_image(scaled)

    # --- 2D 이미지 ---
    def update_recent_image_view(self):
        self._update_viewer_with_qimage(self.scan_viewer,
                                        self.last_color_qimage,
                                        "이미지 없음")

    def update_compare_image_view(self):
        self._update_viewer_with_qimage(self.compare_viewer,
                                        self.compare_color_qimage,
                                        "이미지 없음")

    # --- 뎁스 ---
    def update_recent_depth_view(self):
        if (
            self.show_diff_overlay and
            self.compare_depth_array is not None and
            self.last_depth_array is not None
        ):
            qimg_diff = self.make_diff_depth_qimage()
            if qimg_diff is not None:
                self._update_viewer_with_qimage(self.scan_viewer,
                                                qimg_diff,
                                                "뎁스 데이터 없음")
                return

        self._update_viewer_with_qimage(self.scan_viewer,
                                        self.last_depth_qimage,
                                        "뎁스 데이터 없음")

    def update_compare_depth_view(self):
        self._update_viewer_with_qimage(self.compare_viewer,
                                        self.compare_depth_qimage,
                                        "뎁스 데이터 없음")
        
    def update_recent_pointcloud_diff_view(self):
        """
        정렬된 현재 포인트클라우드에 '거리 기반 색상'을 입혀서 우측 뷰어에 표시.
        """
        res = self.make_pointcloud_diff_for_view()
        if res is None:
            # 실패 시 그냥 일반 포인트 뷰로 fallback
            self.append_log("[WARN] Point diff 뷰 생성 실패 - 일반 포인트 표시로 대체합니다.")
            self.update_recent_pointcloud_view()
            return

        pts_aligned, colors = res
        # ScanViewer 의 show_pointcloud 를 직접 호출
        self._update_viewer_with_pointcloud(
            self.scan_viewer, pts_aligned, colors
        )
    
    def compute_depth_diff_stats(self,
                                 ref_depth: np.ndarray,
                                 cur_depth: np.ndarray):
        """
        Depth 맵 기반 차이 통계 계산.

        return: (mean_diff_cm, max_diff_cm, diff_ratio, threshold_mm) 또는 None
        """
        if ref_depth is None or cur_depth is None:
            return None

        d_ref = ref_depth.astype(np.float32)
        d_cur = cur_depth.astype(np.float32)

        # 두 이미지 크기 맞추기
        h = min(d_ref.shape[0], d_cur.shape[0])
        w = min(d_ref.shape[1], d_cur.shape[1])
        d_ref = d_ref[:h, :w]
        d_cur = d_cur[:h, :w]

        # 유효 픽셀 마스크
        valid_ref = np.isfinite(d_ref) & (d_ref > 0)
        valid_cur = np.isfinite(d_cur) & (d_cur > 0)
        valid_any = valid_ref | valid_cur

        if not np.any(valid_any):
            return None

        both_valid = valid_ref & valid_cur

        diff_mm = np.zeros_like(d_ref, dtype=np.float32)
        diff_mm[both_valid] = np.abs(d_cur[both_valid] - d_ref[both_valid])

        # 한쪽만 유효한 픽셀도 "차이 있음"으로 취급
        threshold_mm = self.get_threshold_mm()
        only_one_valid = valid_any & (~both_valid)
        diff_mm[only_one_valid] = threshold_mm

        # 통계 계산
        valid_vals = diff_mm[valid_any]
        mean_mm = float(valid_vals.mean())
        max_mm = float(valid_vals.max())

        # threshold 이상인 픽셀 비율
        diff_ratio = float(np.mean(valid_vals >= threshold_mm))

        mean_cm = mean_mm / 10.0
        max_cm = max_mm / 10.0

        return mean_cm, max_cm, diff_ratio, threshold_mm

    # ===== 허용 오차 계산 및 표시 =====
    def compute_pointcloud_diff_stats(self,
                                      ref_pts: np.ndarray | None,
                                      cur_pts: np.ndarray | None):
        """
        포인트 클라우드 기반 차이 통계 계산 (부품 위치/자세 보정용).

        ref_pts : 비교 기준 포인트 (Save Compare Data 이후 저장된 것)
        cur_pts : 현재 촬영 포인트

        return: (mean_diff_cm, max_diff_cm, diff_ratio, threshold_mm) 또는 None
        """
        if o3d is None:
            self.append_log("[WARN] open3d 미설치 - 포인트 클라우드 비교를 건너뜁니다.")
            return None

        if ref_pts is None or cur_pts is None:
            return None

        ref = np.asarray(ref_pts, dtype=np.float32)
        cur = np.asarray(cur_pts, dtype=np.float32)
        if ref.size == 0 or cur.size == 0:
            return None

        # open3d 포인트 클라우드로 변환
        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(ref)

        pcd_cur = o3d.geometry.PointCloud()
        pcd_cur.points = o3d.utility.Vector3dVector(cur)

        # ===== 1) 다운샘플 (속도 & 노이즈 감소용) =====
        # 단위가 mm라는 가정 하에 2mm voxel. 필요하면 조정 가능.
        voxel_size = 2.0
        if voxel_size > 0:
            pcd_ref = pcd_ref.voxel_down_sample(voxel_size)
            pcd_cur = pcd_cur.voxel_down_sample(voxel_size)

        if len(pcd_ref.points) == 0 or len(pcd_cur.points) == 0:
            return None

        # ===== 2) ICP로 정합 (cur → ref) =====
        # 정합용 허용 반경 (조금 넉넉하게 3cm)
        align_dist_mm = 30.0
        reg = o3d.pipelines.registration.registration_icp(
            pcd_cur,
            pcd_ref,
            align_dist_mm,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        T = reg.transformation
        self.append_log(
            "[DEBUG] ICP 정합: fitness={:.3f}, rmse={:.3f}, 이동={:.2f}mm".format(
                reg.fitness,
                reg.inlier_rmse,
                np.linalg.norm(T[:3, 3])
            )
        )

        # ===== 3) 정합된 현재 포인트와 기준 포인트 사이 거리 =====
        pcd_cur_aligned = pcd_cur.transform(T)
        dists = np.asarray(pcd_cur_aligned.compute_point_cloud_distance(pcd_ref))
        if dists.size == 0:
            return None

        # 단위: mm 라고 가정
        threshold_mm = self.get_threshold_mm()   # 1cm 이상 차이 나는 포인트 비율
        mean_mm = float(dists.mean())
        max_mm = float(dists.max())
        diff_ratio = float(np.mean(dists >= threshold_mm))

        mean_cm = mean_mm / 10.0
        max_cm = max_mm / 10.0

        return mean_cm, max_cm, diff_ratio, threshold_mm

    def make_diff_depth_qimage(self) -> QImage | None:
        if self.compare_depth_array is None or self.last_depth_array is None:
            return None

        d_ref = self.compare_depth_array.astype(np.float32)
        d_cur = self.last_depth_array.astype(np.float32)

        h = min(d_ref.shape[0], d_cur.shape[0])
        w = min(d_ref.shape[1], d_cur.shape[1])
        d_ref = d_ref[:h, :w]
        d_cur = d_cur[:h, :w]

        # 각쪽에서 유효 픽셀
        valid_ref = np.isfinite(d_ref) & (d_ref > 0)
        valid_cur = np.isfinite(d_cur) & (d_cur > 0)

        valid_any = valid_ref | valid_cur
        if not np.any(valid_any):
            self.append_log("[DEBUG] diff: 유효한 depth 픽셀이 없습니다.")
            return None

        both_valid = valid_ref & valid_cur

        diff_mm = np.zeros_like(d_ref, dtype=np.float32)
        diff_mm[both_valid] = np.abs(d_cur[both_valid] - d_ref[both_valid])

        threshold_mm = self.get_threshold_mm()  # 1 cm
        only_one_valid = valid_any & (~both_valid)
        diff_mm[only_one_valid] = threshold_mm  # 한쪽만 있는 픽셀도 "차이 있음"

        valid_vals = diff_mm[valid_any]
        mean_diff_mm = float(valid_vals.mean())
        max_diff_mm = float(valid_vals.max())

        self.append_log(
            f"[DEBUG] diff 통계: mean={mean_diff_mm:.2f} mm, "
            f"max={max_diff_mm:.2f} mm, thr={threshold_mm:.1f} mm"
        )

        mask_diff = (diff_mm >= threshold_mm) & valid_any

        # === 현재 depth 화면과 동일한 방식으로 크롭 ===
        rgb_uint8 = depth_to_color_image(d_cur)  # 현재 스캔 기준 컬러맵
        mask_valid_obj = np.isfinite(d_cur) & (d_cur > 0)

        if np.any(mask_valid_obj):
            ys, xs = np.where(mask_valid_obj)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            pad_y = int(0.05 * d_cur.shape[0])
            pad_x = int(0.05 * d_cur.shape[1])

            y0 = max(0, y0 - pad_y)
            y1 = min(d_cur.shape[0], y1 + pad_y)
            x0 = max(0, x0 - pad_x)
            x1 = min(d_cur.shape[1], x1 + pad_x)

            rgb_uint8 = rgb_uint8[y0:y1, x0:x1]
            mask_diff = mask_diff[y0:y1, x0:x1]
            valid_any_crop = valid_any[y0:y1, x0:x1]
        else:
            valid_any_crop = valid_any

        rgb_uint8 = np.ascontiguousarray(rgb_uint8)
        h, w, _ = rgb_uint8.shape

        num_valid = int(valid_any_crop.sum())
        num_diff = int(mask_diff.sum())
        self.append_log(
            f"[DEBUG] diff 픽셀 수: valid={num_valid}, diff>=thr={num_diff}"
        )

        if num_diff == 0:
            qimg = QImage(rgb_uint8.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            return qimg

        rgb_uint8[mask_diff] = [255, 255, 255]
        qimg = QImage(rgb_uint8.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return qimg
    
    def make_pointcloud_diff_for_view(self):
        """
        기준 포인트클라우드 vs 현재 포인트클라우드의 차이를
        '정렬된 현재 포인트 + 색상' 형태로 반환.

        return: (pts_aligned, colors) 또는 None
        """
        if o3d is None:
            self.append_log("[WARN] open3d 미설치 - 포인트 diff 뷰를 만들 수 없습니다.")
            return None

        if self.compare_pointcloud is None or self.last_pointcloud is None:
            return None

        ref = np.asarray(self.compare_pointcloud, dtype=np.float32)
        cur = np.asarray(self.last_pointcloud, dtype=np.float32)
        if ref.size == 0 or cur.size == 0:
            return None

        # --- 1) open3d 포인트클라우드로 변환 ---
        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(ref)

        pcd_cur = o3d.geometry.PointCloud()
        pcd_cur.points = o3d.utility.Vector3dVector(cur)

        # --- 2) 다운샘플 (속도 + 노이즈 감소) ---
        voxel_size = 2.0  # 단위가 mm라고 가정, 필요하면 조정
        if voxel_size > 0:
            pcd_ref = pcd_ref.voxel_down_sample(voxel_size)
            pcd_cur = pcd_cur.voxel_down_sample(voxel_size)

        if len(pcd_ref.points) == 0 or len(pcd_cur.points) == 0:
            return None

        # --- 3) ICP 정합 (현재 → 기준) ---
        align_dist_mm = 30.0
        reg = o3d.pipelines.registration.registration_icp(
            pcd_cur,
            pcd_ref,
            align_dist_mm,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        T = reg.transformation
        self.append_log(
            "[DEBUG] (PointDiff) ICP: fitness={:.3f}, rmse={:.3f}, 이동={:.2f}mm".format(
                reg.fitness,
                reg.inlier_rmse,
                np.linalg.norm(T[:3, 3])
            )
        )

        pcd_cur_aligned = pcd_cur.transform(T)

        pts_aligned = np.asarray(pcd_cur_aligned.points)
        if pts_aligned.size == 0:
            return None

        # --- 4) 기준과의 거리 계산 ---
        dists_mm = np.asarray(pcd_cur_aligned.compute_point_cloud_distance(pcd_ref))

        # 시각화용 threshold: 1cm
        threshold_mm = self.get_threshold_mm()

        # 0~threshold 구간만 0~1로 정규화
        dist_clipped = np.clip(dists_mm, 0.0, threshold_mm)
        norm = dist_clipped / (threshold_mm + 1e-6)

        # 가까움(0) = 초록, 임계값 가까움(1) = 빨강 계열
        h = (1.0 - norm) * (1.0 / 3.0)  # 대략 green(120°)~red(0°) 사이
        s = np.ones_like(h, dtype=np.float32)
        v = np.ones_like(h, dtype=np.float32)

        r, g, b = hsv_to_rgb_np(h, s, v)
        colors = np.stack([r, g, b], axis=1).astype(np.float32)

        # threshold 이상 차이나는 포인트는 흰색으로 강조
        mask_large = dists_mm >= threshold_mm
        colors[mask_large] = 1.0  # [1,1,1] = white

        return pts_aligned, colors
    
    def get_threshold_mm(self) -> float:
        """오차 거리(mm) 입력칸에서 값 읽기. 잘못되면 10mm로."""
        default = 10.0
        try:
            text = self.edit_tol_distance.text().strip()
        except AttributeError:
            return default

        if text == "":
            return default

        try:
            val = float(text)
            if val <= 0:
                raise ValueError
            return val
        except ValueError:
            self.append_log("[WARN] 오차 거리(mm) 입력이 잘못되어 기본값 10mm를 사용합니다.")
            self.edit_tol_distance.setText("10")
            return default

    def get_allow_ratio(self) -> float:
        """오차 비율(%) 입력칸에서 값 읽기 → 0~1 비율로 변환. 잘못되면 2%."""
        default = 0.02
        try:
            text = self.edit_tol_ratio.text().strip()
        except AttributeError:
            return default

        if text == "":
            return default

        try:
            val_percent = float(text)
            if val_percent < 0:
                raise ValueError
            return val_percent / 100.0
        except ValueError:
            self.append_log("[WARN] 오차 비율(%) 입력이 잘못되어 기본값 2%를 사용합니다.")
            self.edit_tol_ratio.setText("2")
            return default
        
    def on_apply_tolerance_clicked(self):
        """오차 거리/비율 입력을 현재 데이터에 다시 적용."""
        thr_mm = self.get_threshold_mm()
        ratio = self.get_allow_ratio()  # 0~1

        self.append_log(
            f"[INFO] 허용 오차 기준 재적용: 거리={thr_mm:.1f}mm, 비율={ratio*100:.2f}%"
        )

        # V / X 다시 계산
        self.update_tolerance_display()

        # Diff 표시가 켜져 있으면, 화면도 새 기준으로 다시 갱신
        if self.show_diff_overlay:
            if self.btn_depth.isChecked() and self.last_depth_array is not None:
                self.update_recent_depth_view()

            if (
                self.btn_point.isChecked()
                and self.last_pointcloud is not None
                and self.compare_pointcloud is not None
            ):
                self.update_recent_pointcloud_diff_view()

    def update_tolerance_display(self):
        """
        허용 오차 표시.
        - 가능하면 포인트 클라우드 기준(ICP 정합 후 거리)으로 판정
        - 포인트 클라우드가 없으면 기존 depth 기반 판정 사용
        """

        depth_result = None
        pc_result = None

        # ----- 1) Depth 기반 통계 -----
        if self.compare_depth_array is not None and self.last_depth_array is not None:
            try:
                depth_result = self.compute_depth_diff_stats(
                    self.compare_depth_array, self.last_depth_array
                )
                m_cm, M_cm, ratio, thr_mm = depth_result
                self.append_log(
                    f"[DEBUG] (Depth) 허용오차: mean={m_cm:.2f} cm, "
                    f"max={M_cm:.2f} cm, ratio={ratio*100:.2f}% (thr={thr_mm:.1f} mm)"
                )
            except Exception as e:
                self.append_log(f"[ERROR] Depth 허용 오차 계산 중 오류: {e}")
                depth_result = None

        # ----- 2) Point Cloud 기반 통계 (정합 포함) -----
        if self.compare_pointcloud is not None and self.last_pointcloud is not None:
            try:
                pc_result = self.compute_pointcloud_diff_stats(
                    self.compare_pointcloud, self.last_pointcloud
                )
                if pc_result is not None:
                    m_cm, M_cm, ratio, thr_mm = pc_result
                    self.append_log(
                        f"[DEBUG] (Point) 허용오차: mean={m_cm:.2f} cm, "
                        f"max={M_cm:.2f} cm, ratio={ratio*100:.2f}% (thr={thr_mm:.1f} mm)"
                    )
            except Exception as e:
                self.append_log(f"[ERROR] Point 허용 오차 계산 중 오류: {e}")
                pc_result = None

        # ----- 3) 최종 판정 소스 선택 -----
        # 우선순위: Point → Depth
        result = pc_result if pc_result is not None else depth_result

        if result is None:
            # 비교할 데이터가 없는 경우
            self.label_tol_indicator.setText("-")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:black; font-size:32px; font-weight:bold;"
            )
            return

        mean_cm, max_cm, diff_ratio, threshold_mm = result

        # === 합격 기준 (입력값 기반) ===
        thr_mm = self.get_threshold_mm()
        thr_cm = thr_mm / 10.0
        ratio_limit = self.get_allow_ratio()   # 0~1

        # 1) 평균 차이는 thr_cm 미만
        # 2) thr_mm 이상 차이 나는 포인트 비율이 ratio_limit 미만
        if (mean_cm < thr_cm) and (diff_ratio < ratio_limit):
            self.label_tol_indicator.setText("V")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:lime; font-size:36px; font-weight:bold;"
            )
            src = "Point" if pc_result is not None else "Depth"
            self.append_log(
                f"[DEBUG] 허용오차 판정: OK ({src} 기준, thr={thr_mm:.1f}mm, ratio<{ratio_limit*100:.1f}%)"
            )
        else:
            self.label_tol_indicator.setText("X")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:red; font-size:36px; font-weight:bold;"
            )
            src = "Point" if pc_result is not None else "Depth"
            self.append_log(
                f"[DEBUG] 허용오차 판정: NG ({src} 기준, thr={thr_mm:.1f}mm, ratio>={ratio_limit*100:.1f}%)"
            )

    # ===== 최근 데이터 반환 =====
    def get_latest_pointcloud(self):
        return self.last_pointcloud

    def get_latest_depth(self):
        return self.last_depth_array

    def on_diff_toggle_clicked(self):
        self.show_diff_overlay = self.btn_diff_toggle.isChecked()
        if self.show_diff_overlay:
            self.btn_diff_toggle.setText("Diff 표시 ON")
        else:
            self.btn_diff_toggle.setText("Diff 표시 OFF")

        # 현재 모드에 따라 화면 갱신
        if self.btn_depth.isChecked() and self.last_depth_qimage is not None:
            self.update_recent_depth_view()
        elif self.btn_point.isChecked() and self.last_pointcloud is not None:
            if self.show_diff_overlay and self.compare_pointcloud is not None:
                self.update_recent_pointcloud_diff_view()
            else:
                self.update_recent_pointcloud_view()

    # ===== 창 닫힘 확인 =====
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Exit", "프로그램을 종료하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()