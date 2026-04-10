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

# 깊이 TIFF 읽기용 (둘 중 되는 걸 사용)
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
    QDialog, QListWidget
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
    - Point 모드: pyqtgraph GLViewWidget으로 3D 포인트 클라우드
    - Depths 모드: QLabel에 2D 이미지(QPixmap)
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
        """Depth 모드 등에서 2D 이미지를 표시."""
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

        # 비교 기준 데이터 (좌측)
        self.compare_pointcloud = None
        self.compare_colors = None
        self.compare_depth_qimage: QImage | None = None
        self.compare_depth_array: np.ndarray | None = None

        self.show_diff_overlay = False

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        central.setLayout(main_layout)

        # =======================
        # 상단: 모드 버튼 + Exit
        # =======================
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        self.btn_point = QPushButton("Point")
        self.btn_depth = QPushButton("Depths")

        self.btn_point.setCheckable(True)
        self.btn_depth.setCheckable(True)
        self.btn_point.setChecked(True)

        self.btn_point.clicked.connect(self.on_mode_changed)
        self.btn_depth.clicked.connect(self.on_mode_changed)

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
        self.btn_point.setStyleSheet(btn_style)
        self.btn_depth.setStyleSheet(btn_style)

        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.btn_point)
        group.addButton(self.btn_depth)

        top_layout.addWidget(self.btn_point)
        top_layout.addWidget(self.btn_depth)
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

        # 기존 scan_viewer는 "최근" 뷰로 사용
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

        lbl_tol_desc = QLabel("허용 오차는\n1cm까지")
        lbl_tol_desc.setAlignment(Qt.AlignCenter)
        lbl_tol_desc.setStyleSheet("color:white; font-size:12px;")
        right_side.addWidget(lbl_tol_desc)

        # ===== Diff 오버레이 토글 버튼 추가 =====
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

        right_side.addStretch(1)

        self.btn_refresh = QPushButton("로그 초기화")
        self.btn_refresh.clicked.connect(self.on_refresh_clicked)
        self.btn_refresh.setStyleSheet(
            "QPushButton { background-color:white; color:black; padding:4px; font-size:12px; }"
        )
        # 폭/높이를 조금 제한
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

        # 로그 출력 영역 (흰 박스)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: white; color: black")
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bottom_layout.addWidget(self.log_text, 5)

        # Connect / Scan / Save Compare Data
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

        # 전체 배경 색
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(70, 70, 70))
        self.setPalette(pal)

        # pyqtgraph 경고는 에러 성격이니 로그에 남김
        if pg is None or GLViewWidget is None:
            self.append_log("[ERROR] pyqtgraph / PyOpenGL 미설치. 3D 뷰어를 사용하려면 pip install pyqtgraph PyOpenGL")

    # ===== 로그 도우미 =====
    def append_log(self, text: str):
        """시간을 함께 붙여서 로그 출력."""
        now = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{now}] {text}")

    # ===== Point / Depths 전환 =====
    def on_mode_changed(self):
        """모드 전환 시, 로그는 남기지 않고 화면만 갱신."""
        if self.btn_point.isChecked():
            # Point 모드
            if self.compare_pointcloud is not None:
                self.update_compare_pointcloud_view()
            if self.last_pointcloud is not None:
                self.update_recent_pointcloud_view()
        elif self.btn_depth.isChecked():
            # Depth 모드
            if self.compare_depth_qimage is not None:
                self.update_compare_depth_view()
            if self.last_depth_qimage is not None:
                self.update_recent_depth_view()

    # ===== 로그 초기화 버튼 =====
    def on_refresh_clicked(self):
        """로그 창 초기화 (별도 로그는 남기지 않음)."""
        self.log_text.clear()

    # ===== 카메라 연결 =====
    def connect_camera(self):
        """카메라 연결 (성공 로그는 남기지 않고, 에러만 로그)."""
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
                # 에러 성격
                self.append_log("[ERROR] 검색된 카메라가 없습니다.")
                return

            if result != QDialog.Accepted or dlg.selected_index is None:
                # 사용자가 취소 → 로그 없음
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
        """Scan 버튼: 항상 '최근 데이터(우측)'만 갱신."""
        if USE_REAL_CAMERA and not self.camera_connected:
            self.append_log("[ERROR] 카메라가 연결되지 않았습니다.")
            return

        self.append_log("촬영 실행: Depth + Point 데이터를 스캔합니다.")
        try:
            self.capture_both_depth_and_point()
        except Exception as e:
            self.append_log(f"[ERROR] 캡쳐 중 오류: {e}")
            return

        # 현재 모드에 맞게 '우측'만 업데이트
        if self.btn_point.isChecked():
            if self.last_pointcloud is None:
                self.append_log("[ERROR] 포인트 데이터가 없습니다.")
                return
            self.update_recent_pointcloud_view()
        elif self.btn_depth.isChecked():
            if self.last_depth_qimage is None:
                self.append_log("[ERROR] 뎁스 데이터가 없습니다.")
                return
            self.update_recent_depth_view()

        # 허용 오차 계산 (좌측 기준 데이터가 있을 때만)
        self.update_tolerance_display()

    # ===== 비교 데이터 저장 =====
    def on_save_compare_clicked(self):
        """우측(최근) 데이터를 좌측 비교 기준으로 저장."""
        if self.last_pointcloud is None and self.last_depth_array is None:
            self.append_log("[ERROR] 비교 데이터로 저장할 스캔이 없습니다.")
            return

        # 포인트 클라우드 / 색
        self.compare_pointcloud = None if self.last_pointcloud is None else self.last_pointcloud.copy()
        self.compare_colors = None if self.last_colors is None else self.last_colors.copy()

        # 깊이맵 & 컬러 QImage
        self.compare_depth_array = None if self.last_depth_array is None else self.last_depth_array.copy()
        self.compare_depth_qimage = None if self.last_depth_qimage is None else self.last_depth_qimage.copy()

        # 현재 모드에 맞춰 좌측 뷰 갱신
        if self.btn_point.isChecked():
            self.update_compare_pointcloud_view()
        elif self.btn_depth.isChecked():
            self.update_compare_depth_view()

        self.append_log("비교 데이터 저장 완료.")
        self.update_tolerance_display()

    # ===== 실제 캡쳐 동작 =====
    def capture_both_depth_and_point(self):
        """
        Depth map + Point cloud 둘 다 한 번에 받아서
        '최근 데이터' 변수들(last_*)에 저장.
        """
        if USE_REAL_CAMERA:
            frame3d = Frame3D()
            show_error(self.camera.capture_3d(frame3d))

            # Depth
            depth_map = frame3d.get_depth_map()
            depth_np = depth_map.data().copy().astype(np.float32)
            self.last_depth_array = depth_np

            rgb_uint8 = depth_to_color_image(depth_np)

            # 유효 깊이 영역만 crop (시야 집중)
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

            # Point cloud
            pc = frame3d.get_untextured_point_cloud()
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
            # 샘플 파일 모드
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

    # ===== 포인트 클라우드 뷰 갱신(우측/좌측 공통) =====
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

        # 색 정보 없으면 Z값으로 컬러맵 생성
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

    # ===== 뎁스 뷰 갱신(우측/좌측 공통) =====
    def _update_viewer_with_depth_qimage(self, viewer: ScanViewer, qimg: QImage | None):
        if qimg is None:
            viewer.label.setText("뎁스 데이터 없음")
            viewer.label.show()
            return

        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            viewer.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        viewer.show_image(scaled)

    def update_recent_depth_view(self):
        """
        우측(최근) 뷰 갱신.
        - Diff 토글이 ON이고, 비교 대상/최근 뎁스가 둘 다 있을 때:
            → 차이 부분을 흰색으로 표시한 이미지를 사용
        - 그 외에는:
            → 단순히 최근 스캔 컬러맵을 사용
        """
        # Diff 오버레이 활성화 조건 확인
        if (
            self.show_diff_overlay and
            self.compare_depth_array is not None and
            self.last_depth_array is not None
        ):
            qimg_diff = self.make_diff_depth_qimage()
            if qimg_diff is not None:
                self._update_viewer_with_depth_qimage(self.scan_viewer, qimg_diff)
                return

        # 조건이 안 맞거나 diff 생성 실패 시: 그냥 원본 사용
        self._update_viewer_with_depth_qimage(self.scan_viewer, self.last_depth_qimage)
    def update_compare_depth_view(self):
        self._update_viewer_with_depth_qimage(self.compare_viewer, self.compare_depth_qimage)

    # ===== 허용 오차 계산 및 표시 =====
    def compute_depth_difference_cm(self,
                                    ref_depth: np.ndarray | None,
                                    cur_depth: np.ndarray | None) -> float:
        """
        두 뎁스맵의 평균 절대 오차(cm)를 계산.
        깊이 단위가 '미터'라고 가정하고 100을 곱해 cm로 변환.
        (만약 mm라면 threshold만 10으로 바꿔 쓰면 됨)
        """
        if ref_depth is None or cur_depth is None:
            return float("inf")

        d1 = ref_depth.astype(np.float32)
        d2 = cur_depth.astype(np.float32)

        # 서로 다른 해상도일 경우, 공통 최소 영역만 비교
        h = min(d1.shape[0], d2.shape[0])
        w = min(d1.shape[1], d2.shape[1])
        d1 = d1[:h, :w]
        d2 = d2[:h, :w]

        mask = np.isfinite(d1) & np.isfinite(d2) & (d1 > 0) & (d2 > 0)
        if not np.any(mask):
            return float("inf")

        diff = np.abs(d1 - d2)
        mean_diff = diff[mask].mean()

        # 깊이 단위가 meter라고 보고 cm로 변환
        diff_cm = float(mean_diff * 100.0)
        return diff_cm

    def make_diff_depth_qimage(self) -> QImage | None:
        """
        비교 대상(self.compare_depth_array)을 기준으로,
        최근 스캔(self.last_depth_array)의 차이 부분을 흰색으로 표시한 QImage 생성.

        - 좌측 이미지는 건드리지 않고, 우측(최근) 뷰에만 사용한다.
        - 깊이 단위를 '미터'라고 가정하고, 2mm(0.002m) 이상 차이 나는 곳만 흰색.
          (나중에 숫자를 조절해서 감도 튜닝 가능)
        """
        if self.compare_depth_array is None or self.last_depth_array is None:
            return None

        d_ref = self.compare_depth_array.astype(np.float32)
        d_cur = self.last_depth_array.astype(np.float32)

        # 서로 해상도가 다를 수 있으므로, 공통 영역만 사용
        h = min(d_ref.shape[0], d_cur.shape[0])
        w = min(d_ref.shape[1], d_cur.shape[1])
        d_ref = d_ref[:h, :w]
        d_cur = d_cur[:h, :w]

        # 둘 다 유효한 깊이(>0, finite)인 픽셀만 비교
        mask_valid = (
            np.isfinite(d_ref) & np.isfinite(d_cur) &
            (d_ref > 0) & (d_cur > 0)
        )
        if not np.any(mask_valid):
            self.append_log("[DEBUG] diff: 유효한 depth 픽셀이 없습니다.")
            return None

        # 깊이 차이 (단위: m 라고 가정)
        diff = np.abs(d_cur - d_ref)

        # --- 디버그용 통계 출력 ---
        valid_diff = diff[mask_valid]
        max_diff = float(valid_diff.max())
        mean_diff = float(valid_diff.mean())
        # 임계값: 0.002m = 2mm (필요하면 0.001, 0.005 등으로 조절)
        threshold_m = 0.002
        self.append_log(
            f"[DEBUG] diff 통계: mean={mean_diff:.4f} m, "
            f"max={max_diff:.4f} m, thr={threshold_m:.4f} m"
        )

        mask_diff = mask_valid & (diff >= threshold_m)
        num_valid = int(mask_valid.sum())
        num_diff = int(mask_diff.sum())
        self.append_log(
            f"[DEBUG] diff 픽셀 수: valid={num_valid}, diff>=thr={num_diff}"
        )

        # 차이가 하나도 없으면 원본 컬러맵만 리턴
        if num_diff == 0:
            return depth_to_color_image(d_cur)

        # 기본 배경은 "최근 스캔" 컬러맵
        rgb_uint8 = depth_to_color_image(d_cur)  # (h, w, 3) uint8

        # 크롭은 일단 빼고 전체 이미지에서 바로 마스킹
        rgb_uint8 = np.ascontiguousarray(rgb_uint8)
        rgb_uint8[mask_diff] = [255, 255, 255]

        qimg = QImage(
            rgb_uint8.data, w, h, 3 * w,
            QImage.Format_RGB888
        ).copy()
        return qimg

    def update_tolerance_display(self):
        """좌측 기준 vs 우측 최근 데이터의 오차를 계산해 V / X 표시."""
        if self.compare_depth_array is None or self.last_depth_array is None:
            # 비교 불가 상태
            self.label_tol_indicator.setText("-")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:black; font-size:32px; font-weight:bold;"
            )
            return

        try:
            diff_cm = self.compute_depth_difference_cm(
                self.compare_depth_array, self.last_depth_array
            )
        except Exception as e:
            self.append_log(f"[ERROR] 허용 오차 계산 중 오류: {e}")
            self.label_tol_indicator.setText("?")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:red; font-size:28px; font-weight:bold;"
            )
            return

        if diff_cm < 1.0:
            self.label_tol_indicator.setText("V")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:lime; font-size:36px; font-weight:bold;"
            )
        else:
            self.label_tol_indicator.setText("X")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:red; font-size:36px; font-weight:bold;"
            )

    # ===== 최근 데이터 반환 (필요하면 외부에서 사용) =====
    def get_latest_pointcloud(self):
        return self.last_pointcloud

    def get_latest_depth(self):
        return self.last_depth_array
    
    def on_diff_toggle_clicked(self):
        """
        Diff 표시 ON/OFF 토글.
        - 플래그만 바꾸고, Depth 모드 + 최근 뎁스가 있으면 우측 뷰만 다시 그린다.
        - 비교 대상(좌측)은 항상 원본 그대로 유지.
        """
        self.show_diff_overlay = self.btn_diff_toggle.isChecked()
        if self.show_diff_overlay:
            self.btn_diff_toggle.setText("Diff 표시 ON")
        else:
            self.btn_diff_toggle.setText("Diff 표시 OFF")

        # Depth 모드에서만 의미 있으니, 그때만 즉시 리프레시
        if self.btn_depth.isChecked() and self.last_depth_qimage is not None:
            self.update_recent_depth_view()

    

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