import os
import sys
import numpy as np

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
#   - False : sample 폴더의 PLY / TIFF 파일을 사용 (지금 테스트용 모드)
#   - True  : 아래 connect_camera / capture_both_depth_and_point 안의
#             SDK 코드(MechEyeAPI 등)를 타게 됨.
# --------------------------------------------------------------------
USE_REAL_CAMERA = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_PLY_PATH = os.path.join(BASE_DIR, "sample", "sample_point.ply")
SAMPLE_TIFF_PATH = os.path.join(BASE_DIR, "sample", "sample_depth.tiff")

# =========================================================
# ★ 다른 3D 카메라 SDK 를 쓰고 싶다면, 이 부분에 import 추가
#    예)
#    from My3DCamSDK import MyCamera, MyFrame
#
#  지금은 MechEye Python SDK 를 사용 중:
#  - from mecheye.shared import *
#  - from mecheye.area_scan_3d_camera import *
#  - from mecheye.area_scan_3d_camera_utils import *
# =========================================================


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

    # ★ NaN, ±inf 정리해두면 cast 관련 경고를 줄일 수 있음
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


    # 채널이 여러 개면 첫 채널만 사용 (대부분 1채널 TIFF일 것)
    if depth.ndim == 3:
        depth = depth[..., 0]

    mask = np.isfinite(depth) & (depth > 0)
    if not np.any(mask):
        # 유효한 깊이가 없으면 그냥 검정 이미지
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
    # h = 0 (red) ~ 2/3 (blue) 범위 사용
    h = (norm) * (2.0 / 3.0)
    s = np.ones_like(h)
    v = np.ones_like(h)

    r, g, b = hsv_to_rgb_np(h, s, v)
    rgb = np.stack([r, g, b], axis=-1)  # [0,1]

    rgb_uint8 = (rgb * 255.0).astype(np.uint8)
    # 마스크 밖(깊이 0 등)은 검정
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

        # 카메라 리스트 박스
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        main_layout.addWidget(self.list_widget)

        # === 리스트 채우기 ===
        if not camera_infos:
            # 카메라가 하나도 없을 때
            self.list_widget.addItem("연결된 카메라가 없습니다.")
            self.list_widget.setEnabled(False)   # 선택 불가
        else:
            for i, info in enumerate(camera_infos):
                try:
                    text = f"{i}: {info.modelName} | IP: {info.ipAddress} | SN: {info.serialNumber}"
                except Exception:
                    text = f"{i}: {info}"
                self.list_widget.addItem(text)

        # 하단 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_connect = QPushButton("연결")
        self.btn_connect.setFixedWidth(100)
        self.btn_connect.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color:white; padding:6px; }"
        )
        self.btn_connect.clicked.connect(self.on_connect_clicked)

        # 카메라 없으면 연결 버튼도 비활성화
        if not camera_infos:
            self.btn_connect.setEnabled(False)

        btn_layout.addWidget(self.btn_connect)
        main_layout.addLayout(btn_layout)

        # 더블클릭으로도 선택 가능하게 (카메라 있을 때만 의미 있음)
        self.list_widget.itemDoubleClicked.connect(self.on_connect_clicked)

    def on_connect_clicked(self, *args):
        row = self.list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "선택 필요", "연결할 카메라를 선택하세요.")
            return
        self.selected_index = row
        self.accept()  # 다이얼로그 종료 + exec()에 Accepted 리턴


class ScanViewer(QWidget):
    """
    왼쪽 스캔 데이터 출력 영역.
    - Point 모드: pyqtgraph GLViewWidget으로 3D 포인트 클라우드
    - Depths 모드: QLabel에 2D 이미지(QPixmap)
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

        # 이미지/텍스트용 라벨 (Depth 모드나 에러 표시용)
        self.label = QLabel("스캔 데이터 출력")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 18px;")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._layout.addWidget(self.label)

        # 3D 뷰어(pyqtgraph)는 필요할 때 생성
        self.gl_view = None
        self.gl_scatter = None

        # 배경을 검은색으로
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(0, 0, 0))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    # --------- 3D 뷰어 관련 ---------
    def ensure_gl_view(self):
        """GLViewWidget이 없다면 생성하고, 라벨 대신 보이게 한다."""
        if GLViewWidget is None or GLScatterPlotItem is None:
            return False

        if self.gl_view is None:
            self.gl_view = GLViewWidget()
            # 배경 검은색
            self.gl_view.setBackgroundColor('k')
            # 기본 카메라 파라미터
            self.gl_view.opts['fov'] = 60
            self.gl_view.opts['elevation'] = 25
            self.gl_view.opts['azimuth'] = 45
            self.gl_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._layout.addWidget(self.gl_view)

        # 모드 전환 시: 라벨 숨기고 3D 뷰어 보이기
        self.label.hide()
        self.gl_view.show()
        return True

    def show_pointcloud(self, points: np.ndarray, colors: np.ndarray | None = None):
        """3D 포인트 클라우드를 pyqtgraph GLViewWidget에 그리기."""
        if not self.ensure_gl_view():
            # pyqtgraph 미설치 시 안내
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

        # --- 1) 포인트 수가 너무 많으면 다운샘플 (선명도 + 성능 ↑) ---
        max_points = 150_000
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
            if colors is not None:
                colors = colors[idx]

        # --- 2) 센터 맞추기 ---
        center = pts.mean(axis=0)
        pts_centered = pts - center

        # --- 3) 색 정보 처리 (0~255 -> 0~1, 그리고 RGBA로 확장) ---
        if colors is not None:
            cols = colors.astype(np.float32)
            if cols.max() > 1.0:
                cols = cols / 255.0  # [0,255] → [0,1]
        else:
            # 흰색 포인트
            cols = np.ones((pts.shape[0], 3), dtype=np.float32)

        # (N,3) -> (N,4) 로 알파=1.0 추가
        if cols.ndim == 2 and cols.shape[1] == 3:
            alpha = np.ones((cols.shape[0], 1), dtype=np.float32)
            cols_rgba = np.concatenate([cols, alpha], axis=1)
        else:
            cols_rgba = cols

        # --- 4) 카메라 distance를 포인트 반경 기준으로 맞춤 ---
        radius = float(np.linalg.norm(pts_centered, axis=1).max())
        if radius > 0:
            # 반경의 2~3배 정도 거리에 카메라 위치
            self.gl_view.opts['distance'] = radius * 2.5

        # --- 5) GLScatterPlotItem 생성/업데이트 ---
        point_size = 2.0  # px 단위, 필요하면 1.0~3.0 사이에서 조절

        if self.gl_scatter is None:
            self.gl_scatter = GLScatterPlotItem(
                pos=pts_centered,
                color=cols_rgba,
                size=point_size,
                pxMode=True,
                glOptions='opaque',  # 색이 번지지 않도록 opaque 모드
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

        # 카메라 center를 (0,0,0)으로
        if pg is not None:
            self.gl_view.opts['center'] = pg.Vector(0, 0, 0)

        self.gl_view.show()

    # --------- 2D 이미지(Depth) 관련 ---------
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
        self.resize(940, 620)

        # -------------------------------------------------
        # 카메라 핸들 & 상태 변수
        # -------------------------------------------------
        # ★ 실제 SDK의 카메라 객체를 여기에 보관
        #   (지금은 MechEye 의 Camera 클래스를 사용 중)
        self.camera = None
        self.camera_connected = False

        # 마지막 캡쳐 데이터 캐시
        self.last_pointcloud = None     # np.ndarray (N,3)
        self.last_colors = None
        self.last_depth_qimage: QImage | None = None
        self.last_depth_array: np.ndarray | None = None   # ★ 원본 depth map

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        central.setLayout(main_layout)

        # =======================
        # 왼쪽 영역
        # =======================
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)

        # 상단 Point / Depths 버튼 줄
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(8)

        self.btn_point = QPushButton("Point")
        self.btn_depth = QPushButton("Depths")

        # 토글 버튼처럼 동작하게
        self.btn_point.setCheckable(True)
        self.btn_depth.setCheckable(True)
        self.btn_point.setChecked(True)

        self.btn_point.clicked.connect(self.on_mode_changed)
        self.btn_depth.clicked.connect(self.on_mode_changed)

        # 색감
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

        mode_layout.addWidget(self.btn_point)
        mode_layout.addWidget(self.btn_depth)

        left_layout.addLayout(mode_layout)

        # 스캔 뷰어(검은 박스, 내부에서 3D/2D 전환)
        self.scan_viewer = ScanViewer()
        self.scan_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.scan_viewer)

        main_layout.addLayout(left_layout, 3)  # 비율상 왼쪽을 좀 더 크게

        # =======================
        # 오른쪽 영역
        # =======================
        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)

        # 상단: 새로고침/회전 버튼 + Exit 버튼
        top_right_layout = QHBoxLayout()
        top_right_layout.setSpacing(8)

        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setFixedWidth(40)
        self.btn_refresh.clicked.connect(self.on_refresh_clicked)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setStyleSheet(
            "QPushButton { background-color: red; color: white; font-weight: bold; padding: 6px; }"
        )
        self.btn_exit.setFixedWidth(80)
        self.btn_exit.clicked.connect(self.close)

        top_right_layout.addWidget(self.btn_refresh)
        top_right_layout.addStretch(1)
        top_right_layout.addWidget(self.btn_exit)

        right_layout.addLayout(top_right_layout)

        # 로그 출력 영역 (흰 박스)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: white; color: black")
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.log_text)

        # 하단 Connect / Scan 버튼
        bottom_btn_layout = QVBoxLayout()
        bottom_btn_layout.setSpacing(8)

        self.btn_connect = QPushButton("Connect")
        self.btn_scan = QPushButton("Scan")

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
        self.btn_connect.setStyleSheet(bottom_btn_style)
        self.btn_scan.setStyleSheet(bottom_btn_style)

        self.btn_connect.clicked.connect(self.connect_camera)
        self.btn_scan.clicked.connect(self.scan_pointcloud)
        self.btn_scan.setEnabled(False)   # 처음에는 비활성화

        bottom_btn_layout.addWidget(self.btn_connect)
        bottom_btn_layout.addWidget(self.btn_scan)

        right_layout.addLayout(bottom_btn_layout)

        main_layout.addLayout(right_layout, 2)

        self.append_log("프로그램 시작. Connect 버튼으로 카메라를 연결하세요.")
        if pg is None or GLViewWidget is None:
            self.append_log("[WARN] pyqtgraph / PyOpenGL 미설치. 3D 뷰어 사용 시 설치 필요: pip install pyqtgraph PyOpenGL")

        # 전체 배경 색
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(70, 70, 70))
        self.setPalette(pal)

    # ===== 로그 도우미 =====
    def append_log(self, text: str):
        self.log_text.append(text)

    # ===== Point / Depths 전환 =====
    def on_mode_changed(self):
        if self.btn_point.isChecked():
            self.append_log("[MODE] Point 모드 선택")
            if self.last_pointcloud is not None:
                self.update_pointcloud_view()
        elif self.btn_depth.isChecked():
            self.append_log("[MODE] Depths 모드 선택")
            if self.last_depth_qimage is not None:
                self.load_depth_sample(redraw_only=True)

    # ===== 새로고침 버튼 =====
    def on_refresh_clicked(self):
        """Refresh 버튼: 로그 창 초기화."""
        self.log_text.clear()
        # 초기 안내 문구 하나 넣고 싶으면:
        self.append_log("로그를 초기화했습니다.")

    # ===== 카메라 연결 =====
    def connect_camera(self):
        """카메라 없는 환경에서는 더미 연결만 수행."""
        if not USE_REAL_CAMERA:
            # ---------------------------------------------------------
            # ★ 지금은 샘플 PLY/TIFF만 쓸 예정이므로,
            #   실제 카메라 연결은 건너뛰고 Scan 버튼만 활성화.
            #   나중에 실제 장비 테스트할 때는 USE_REAL_CAMERA=True 로.
            # ---------------------------------------------------------
            self.append_log("[INFO] (더미) 카메라 연결을 생략합니다. 샘플 파일 모드로 동작합니다.")
            self.camera_connected = True
            self.btn_scan.setEnabled(True)
            return

        # === 실제 카메라 있는 환경 ===
        try:
            self.append_log("Discovering all available cameras...")

            # ---------------------------------------------------------
            # ★ (1) SDK가 제공하는 '카메라 검색' API 사용 부분
            #     - MechEye: Camera.discover_cameras()
            #     - 다른 SDK: list_devices(), enumerate_cameras() 등
            #   필요하면 아래 한 줄을 해당 함수로 교체해도 됨.
            # ---------------------------------------------------------
            camera_infos = Camera.discover_cameras()

            dlg = CameraSelectDialog(camera_infos, self)
            result = dlg.exec()

            if not camera_infos:
                self.append_log("검색된 카메라가 없습니다.")
                return

            if result != QDialog.Accepted or dlg.selected_index is None:
                self.append_log("Camera selection canceled.")
                return

            idx = dlg.selected_index
            self.append_log(f"카메라 index {idx} 선택. 연결 시도 중...")

            # ---------------------------------------------------------
            # ★ (2) SDK의 '카메라 객체 생성 + 선택된 장치에 연결' 부분
            #     - self.camera = Camera()
            #     - error_status = self.camera.connect(camera_infos[idx])
            #   다른 SDK를 쓰면, 여기만 해당 방식으로 바꿔주면 된다.
            # ---------------------------------------------------------
            self.camera = Camera()
            error_status = self.camera.connect(camera_infos[idx])

            if not error_status.is_ok():
                try:
                    show_error(error_status)
                except Exception:
                    pass

                msg = f"카메라 연결 실패: {error_status}"
                self.append_log("[ERROR] " + msg)
                self.camera_connected = False
                self.btn_scan.setEnabled(False)
                return

            self.camera_connected = True
            self.btn_scan.setEnabled(True)
            self.append_log("Connected to the camera successfully.")

        except Exception as e:
            self.append_log(f"[EXCEPTION] 카메라 연결 중 오류: {e}")
            self.camera_connected = False
            self.btn_scan.setEnabled(False)

    # ===== 스캔 =====
    def scan_pointcloud(self):
        # 1) 항상 depth + point 둘 다 캡쳐해서 last_*에 저장
        self.append_log("[INFO] 스캔 시작 (Depth + Point 동시 캡쳐)...")
        try:
            self.capture_both_depth_and_point()
        except Exception as e:
            self.append_log(f"[ERROR] 캡쳐 중 오류: {e}")
            return

        # 2) 현재 모드에 맞게 화면에만 뿌려줌 (데이터는 이미 last_*에 있음)
        if self.btn_point.isChecked():
            if self.last_pointcloud is None:
                self.append_log("[WARN] 포인트 데이터가 없습니다.")
                return
            self.update_pointcloud_view()
        elif self.btn_depth.isChecked():
            if self.last_depth_qimage is None:
                self.append_log("[WARN] 뎁스 데이터가 없습니다.")
                return
            pixmap = QPixmap.fromImage(self.last_depth_qimage)
            scaled = pixmap.scaled(
                self.scan_viewer.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.scan_viewer.show_image(scaled)

    def capture_both_depth_and_point(self):
        """
        Scan 버튼이 눌렸을 때 실제로 하는 일:
        - Depth map + Point cloud 둘 다 한 번에 받아와서
          last_depth_array, last_depth_qimage, last_pointcloud, last_colors 를 채운다.
        - 화면에는 아직 아무것도 안 그림 (그릴지는 호출한 쪽이 결정).
        """
        if USE_REAL_CAMERA:
            # ===== 실제 카메라에서 캡쳐 =====

            # -----------------------------------------------------
            # ★ (1) SDK에서 제공하는 3D 프레임 타입 생성
            #     - MechEye: Frame3D()
            #     - 다른 SDK: Frame, CaptureResult 등
            # -----------------------------------------------------
            frame3d = Frame3D()

            # -----------------------------------------------------
            # ★ (2) 실제 한 번 촬영(capture)하는 부분
            #     - self.camera.capture_3d(frame3d)
            #     - 또는 self.camera.grab_frame(frame3d) 등
            #   -> 실패 시 show_error 로 에러 팝업도 띄움.
            # -----------------------------------------------------
            show_error(self.camera.capture_3d(frame3d))

            # ---------------- Depth ----------------
            # ★ (3) 프레임에서 depth map 꺼내기
            #     - get_depth_map() / get_range_image() 등
            depth_map = frame3d.get_depth_map()
            depth_np = depth_map.data().copy().astype(np.float32)
            self.last_depth_array = depth_np   # ★ 원본은 그대로 저장해둠

            # 1) 전체 컬러맵
            rgb_uint8 = depth_to_color_image(depth_np)

            # 2) 유효 깊이 영역만 bounding box로 crop 해서 물체를 키워서 보이게
            mask_valid = np.isfinite(depth_np) & (depth_np > 0)
            if np.any(mask_valid):
                ys, xs = np.where(mask_valid)
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1

                # 살짝 여유 패딩(화면이 너무 꽉 차지 않게)
                pad_y = int(0.05 * depth_np.shape[0])
                pad_x = int(0.05 * depth_np.shape[1])

                y0 = max(0, y0 - pad_y)
                y1 = min(depth_np.shape[0], y1 + pad_y)
                x0 = max(0, x0 - pad_x)
                x1 = min(depth_np.shape[1], x1 + pad_x)

                rgb_uint8 = rgb_uint8[y0:y1, x0:x1]

            # 3) QImage용으로 메모리 연속 보장
            rgb_uint8 = np.ascontiguousarray(rgb_uint8)

            h, w, _ = rgb_uint8.shape
            qimg = QImage(
                rgb_uint8.data, w, h, 3 * w,
                QImage.Format_RGB888
            ).copy()
            self.last_depth_qimage = qimg

            # ---------------- Point Cloud ----------------
            # ★ (4) 포인트 클라우드 꺼내기
            #     - get_untextured_point_cloud()
            #     - 또는 get_point_cloud() 등
            pc = frame3d.get_untextured_point_cloud()
            pc_np = pc.data().copy()   # 보통 (H,W,3) 또는 (N,3)

            pts = pc_np
            if pts.ndim == 3:
                # (H,W,3) → (N,3) 로 펼치기
                pts = pts.reshape(-1, 3)

            # 유효 포인트만 사용 (NaN 제거, Z>0 필터)
            mask = np.isfinite(pts).all(axis=1)
            mask &= (pts[:, 2] > 0)   # Z>0 만 사용 (원하면 빼도 됨)
            pts = pts[mask]

            self.last_pointcloud = pts
            self.last_colors = None   # 색이 따로 없다고 가정 (필요하면 SDK에서 색도 꺼내기)

            # -----------------------------------------------------
            # ★ (5) 여기에서 파일 저장을 하고 싶으면 추가
            #     예)
            #       np.save("last_depth.npy", self.last_depth_array)
            #       pcd = o3d.geometry.PointCloud()
            #       pcd.points = o3d.utility.Vector3dVector(self.last_pointcloud)
            #       o3d.io.write_point_cloud("last_pointcloud.ply", pcd)
            # -----------------------------------------------------

        else:
            # ===== 샘플 파일 모드 =====
            # 1) PLY → last_pointcloud / last_colors 채우기 (렌더 X)
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

            # 2) TIFF → last_depth_array / last_depth_qimage 채우기 (렌더 X)
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

    # 아래 두 함수(load_ply_sample, load_depth_sample)는
    # 샘플 파일 모드에서만 직접 호출되며,
    # 실제 카메라 모드에서는 capture_both_depth_and_point() 를 사용.
    def load_ply_sample(self):
        path = SAMPLE_PLY_PATH
        self.append_log(f"[INFO] PLY 샘플 로드: {path}")

        if o3d is None:
            self.append_log("[WARN] open3d 미설치. 'pip install open3d' 필요.")
            self.scan_viewer.label.setText("open3d 필요 (Point 모드)")
            return

        pcd_raw = o3d.io.read_point_cloud(path)
        pts_np = np.asarray(pcd_raw.points)
        self.append_log(f"[DEBUG] 원본 포인트 수: {pts_np.shape[0]}")

        mask = np.isfinite(pts_np).all(axis=1)
        pts = pts_np[mask]
        self.append_log(f"[DEBUG] 유효 포인트 수(Non-NaN): {pts.shape[0]}")

        if pts.shape[0] == 0:
            self.append_log("[ERROR] 유효한 포인트가 하나도 없습니다.")
            self.scan_viewer.label.setText("유효 포인트 없음")
            return

        self.last_pointcloud = pts

        if pcd_raw.has_colors():
            cols_np = np.asarray(pcd_raw.colors)
            self.last_colors = cols_np[mask]
        else:
            self.last_colors = None

        num_points = pts.shape[0]
        min_xyz = pts.min(axis=0)
        max_xyz = pts.max(axis=0)

        self.append_log(f"[OK] PLY 로드 성공. Point 수: {num_points}")
        self.append_log(f" - Min: {min_xyz}")
        self.append_log(f" - Max: {max_xyz}")

        self.update_pointcloud_view()

    def load_depth_sample(self, redraw_only=False):
        """샘플 TIFF 뎁스 이미지를 레인보우 컬러로 변환해서 왼쪽 뷰에 표시."""
        path = SAMPLE_TIFF_PATH

        # 이미 계산해 둔 컬러 QImage가 있고, 단순 리프레시면 그대로 사용
        if redraw_only:
            if self.last_depth_qimage is None:
                return
            base_qimg = self.last_depth_qimage
        else:
            self.append_log(f"[INFO] TIFF 샘플 로드: {path}")

            depth_np = None

            # 1) OpenCV가 있으면 그걸로 시도
            if cv2 is not None:
                depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if depth is not None:
                    depth_np = depth.astype(np.float32)

            # 2) 실패하면 PIL 시도
            if depth_np is None and Image is not None:
                try:
                    img = Image.open(path)
                    depth_np = np.array(img).astype(np.float32)
                except Exception:
                    depth_np = None

            # 3) 둘 다 없으면 QPixmap 직통 (흑백 fallback)
            if depth_np is None:
                self.append_log("[WARN] OpenCV/PIL 미설치 또는 TIFF 읽기 실패. "
                                "깊이 컬러맵 대신 원본 이미지를 표시합니다.")
                pixmap = QPixmap(path)
                if pixmap.isNull():
                    self.append_log("[ERROR] TIFF 로드 실패.")
                    self.scan_viewer.label.setText("TIFF 로드 오류")
                    return
                scaled = pixmap.scaled(
                    self.scan_viewer.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.scan_viewer.show_image(scaled)
                # fallback일 때는 last_depth_qimage는 None으로 남김
                return

            # 깊이 → 컬러맵
            rgb_uint8 = depth_to_color_image(depth_np)

            # ★ 유효 깊이 영역만 crop (실카메라와 동일한 로직)
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

            # ★ QImage용 연속 버퍼 확보
            rgb_uint8 = np.ascontiguousarray(rgb_uint8)

            h, w, _ = rgb_uint8.shape
            qimg = QImage(
                rgb_uint8.data,
                w,
                h,
                3 * w,
                QImage.Format_RGB888
            ).copy()  # 버퍼 독립
            self.last_depth_qimage = qimg
            base_qimg = qimg
            self.append_log("[OK] 깊이 컬러맵 이미지 생성 완료.")

        # ScanViewer 크기에 맞게 스케일 후 표시
        pixmap = QPixmap.fromImage(base_qimg)
        scaled = pixmap.scaled(
            self.scan_viewer.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.scan_viewer.show_image(scaled)

    def update_pointcloud_view(self):
        """self.last_pointcloud를 3D 뷰어에 갱신."""
        if self.last_pointcloud is None:
            return

        pts = self.last_pointcloud
        cols = self.last_colors

        if o3d is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if cols is not None:
                # 이미 색 정보가 있으면 그대로 사용 (샘플 PLY 등)
                pcd.colors = o3d.utility.Vector3dVector(cols)

            # 센터 맞추기 + 기본 뒤집기(위아래)
            center = pcd.get_center()
            pcd.translate(-center)
            base_R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            pcd.rotate(base_R, center=(0, 0, 0))

            pts_vis = np.asarray(pcd.points)
            if pcd.has_colors():
                cols_vis = np.asarray(pcd.colors)
            else:
                cols_vis = None
        else:
            pts_vis = pts
            cols_vis = cols

        # -------------------------------------------------
        # ★ 색 정보가 없다면(Z값 기준) 컬러맵을 자동으로 생성
        #    - Depth 컬러맵과 같은 스타일: 가까운(작은 Z) = 빨강, 먼(큰 Z) = 파랑
        # -------------------------------------------------
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

                # depth_to_color_image와 동일한 HSV 매핑:
                # h = 0 (red) ~ 2/3 (blue)
                h = (1 - norm) * (2.0 / 3.0)
                s = np.ones_like(h, dtype=np.float32)
                v = np.ones_like(h, dtype=np.float32)

                r, g, b = hsv_to_rgb_np(h, s, v)
                cols_vis = np.stack([r, g, b], axis=-1).astype(np.float32)
            else:
                cols_vis = None

        # pyqtgraph 뷰어에 전달
        self.scan_viewer.show_pointcloud(pts_vis, cols_vis)


    def get_latest_pointcloud(self):
        """가장 최근 Scan에서 캡쳐한 포인트 클라우드 (N,3) 반환."""
        return self.last_pointcloud

    def get_latest_depth(self):
        """가장 최근 Scan에서 캡쳐한 raw depth map (H,W 또는 H,W,1) 반환."""
        return self.last_depth_array
    
    # 창 닫힐 때 확인(선택 사항)
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