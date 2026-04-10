import os
import subprocess
import sys
import shutil
import re
import numpy as np
import json
import time
from typing import Union
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

from PySide6.QtCore import Qt, QObject, Signal, QThread, QEvent, QTimer, QPoint
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QSizePolicy, QButtonGroup, QMessageBox,
    QDialog, QListWidget, QLineEdit, QCheckBox,
    QGridLayout, QFrame
)
from PySide6.QtGui import QPalette, QColor, QPixmap, QImage, QGuiApplication, QCursor

# --------------------------------------------------------------------
# ★ 실제 카메라를 쓸 때는 True 로 바꾸기.
# --------------------------------------------------------------------
USE_REAL_CAMERA = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_PLY_PATH = os.path.join(BASE_DIR, "sample", "sample_point.ply")
SAMPLE_TIFF_PATH = os.path.join(BASE_DIR, "sample", "sample_depth.tiff")
SAMPLE_IMG_PATH = os.path.join(BASE_DIR, "sample", "sampleimg.png")

# 결과 저장 폴더(최근 스캔 저장)
RESULT_DIR = os.path.join(BASE_DIR, "result", "objects_ply_data", "2025")

# ★ 비교데이터 저장 폴더(요청사항)
COMPARE_DIR = os.path.join(BASE_DIR, "comparedata")


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


def sanitize_folder_component(text: str) -> str:
    """폴더/파일명에 위험한 문자 제거."""
    if text is None:
        return ""
    text = text.strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def list_compare_folders() -> list[str]:
    """COMPARE_DIR 안의 폴더 목록을 반환(정렬)."""
    if not os.path.isdir(COMPARE_DIR):
        return []
    items = []
    for name in os.listdir(COMPARE_DIR):
        p = os.path.join(COMPARE_DIR, name)
        if os.path.isdir(p):
            items.append(name)
    items.sort()
    return items


def find_folder_by_number(num_text: str) -> str | None:
    """번호로 시작하는 폴더(예: 12_XXX) 찾기."""
    num_text = (num_text or "").strip()
    if num_text == "":
        return None
    folders = list_compare_folders()
    prefix = f"{num_text}_"
    for f in folders:
        if f.startswith(prefix):
            return f
    return None


# ============================================================
# Hangul Composer (초/중/종성 조합) + Virtual Keyboard
# ============================================================

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

# 복합 모음/받침 규칙
JUNG_COMBINE = {
    ("ㅗ", "ㅏ"): "ㅘ",
    ("ㅗ", "ㅐ"): "ㅙ",
    ("ㅗ", "ㅣ"): "ㅚ",
    ("ㅜ", "ㅓ"): "ㅝ",
    ("ㅜ", "ㅔ"): "ㅞ",
    ("ㅜ", "ㅣ"): "ㅟ",
    ("ㅡ", "ㅣ"): "ㅢ",
}
JUNG_SPLIT = {v: k for k, v in JUNG_COMBINE.items()}

JONG_COMBINE = {
    ("ㄱ", "ㅅ"): "ㄳ",
    ("ㄴ", "ㅈ"): "ㄵ",
    ("ㄴ", "ㅎ"): "ㄶ",
    ("ㄹ", "ㄱ"): "ㄺ",
    ("ㄹ", "ㅁ"): "ㄻ",
    ("ㄹ", "ㅂ"): "ㄼ",
    ("ㄹ", "ㅅ"): "ㄽ",
    ("ㄹ", "ㅌ"): "ㄾ",
    ("ㄹ", "ㅍ"): "ㄿ",
    ("ㄹ", "ㅎ"): "ㅀ",
    ("ㅂ", "ㅅ"): "ㅄ",
}
JONG_SPLIT = {v: k for k, v in JONG_COMBINE.items()}


def is_ja(c: str) -> bool:
    return c in CHO or c in JONG[1:]  # 종성 자모 포함


def is_mo(c: str) -> bool:
    return c in JUNG


def compose_syllable(cho: str, jung: str, jong: str = "") -> str:
    if cho not in CHO or jung not in JUNG:
        return ""
    ci = CHO.index(cho)
    ji = JUNG.index(jung)
    ti = JONG.index(jong) if jong in JONG else 0
    code = 0xAC00 + (ci * 21 * 28) + (ji * 28) + ti
    return chr(code)


class HangulComposer:
    """
    아주 기본적인 한글 조합 상태 머신.
    - 조합중인 글자(초/중/종)를 상태로 갖고 있다가 commit_text / flush로 확정한다.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.cho = ""
        self.jung = ""
        self.jong = ""

    def has_composing(self) -> bool:
        return self.cho != "" or self.jung != "" or self.jong != ""

    def composing_text(self) -> str:
        if self.cho and self.jung:
            return compose_syllable(self.cho, self.jung, self.jong)
        # 초성만 있는 상태면 자모 그대로 보이게
        if self.cho and not self.jung:
            return self.cho
        # 모음만 단독 입력되는 경우
        if self.jung and not self.cho:
            return self.jung
        return ""

    def input_char(self, c: str):
        """
        입력 문자 c를 받아서,
        - (commit_string, composing_string) 형태로 반환
        commit_string은 즉시 확정해서 QLineEdit에 넣을 텍스트,
        composing_string은 조합중으로 보일 텍스트(마지막에 덮어쓰기용)
        """
        commit = ""

        # 공백/기타는 조합 확정 후 그대로 커밋
        if not (is_ja(c) or is_mo(c)):
            if self.has_composing():
                commit += self.composing_text()
                self.reset()
            commit += c
            return commit, ""

        # --------------------
        # 모음 입력
        # --------------------
        if is_mo(c):
            # (1) 초성 없이 모음부터 들어오면: 모음 단독
            if self.cho == "" and self.jung == "":
                self.jung = c
                return "", self.composing_text()

            # (2) 초성만 있는 상태 -> 중성 채우기
            if self.cho != "" and self.jung == "":
                self.jung = c
                return "", self.composing_text()

            # (3) 중성이 이미 있는데 모음이 또 오면 복합모음 시도
            if self.jung != "":
                # 종성이 있으면: 종성을 다음 글자 초성으로 넘기는 케이스 처리
                if self.jong != "":
                    # 종성이 복합받침이면 분리
                    if self.jong in JONG_SPLIT:
                        a, b = JONG_SPLIT[self.jong]
                        # 앞 글자에는 a만 종성으로 남기고, b는 다음 글자 초성으로
                        prev = compose_syllable(self.cho, self.jung, a)
                        self.cho = b
                        self.jung = c
                        self.jong = ""
                        commit += prev
                        return commit, self.composing_text()
                    else:
                        # ✅ 단일 종성: 앞 글자는 종성 없이 확정하고,
                        # ✅ 종성을 다음 글자 초성으로 넘김
                        prev = compose_syllable(self.cho, self.jung, "")
                        commit += prev
                        self.cho = self.jong
                        self.jung = c
                        self.jong = ""
                        return commit, self.composing_text()

                # 종성 없으면 복합모음 결합
                key = (self.jung, c)
                if key in JUNG_COMBINE:
                    self.jung = JUNG_COMBINE[key]
                    return "", self.composing_text()
                else:
                    # 복합모음 불가 -> 이전 글자 확정 후 새 모음 시작
                    commit += self.composing_text()
                    self.reset()
                    self.jung = c
                    return commit, self.composing_text()

        # --------------------
        # 자음 입력
        # --------------------
        if is_ja(c):
            # (1) 아무것도 없으면 초성으로 시작
            if self.cho == "" and self.jung == "":
                self.cho = c
                return "", self.composing_text()

            # (2) 초성만 있는데 자음 또 오면: 된소리/겹자음 초성은 여기선 단순 처리(이전 확정 후 새 초성)
            if self.cho != "" and self.jung == "":
                commit += self.cho
                self.cho = c
                return commit, self.composing_text()

            # (3) 초+중 있는 상태에서 자음: 종성으로 들어감
            if self.cho != "" and self.jung != "":
                # 종성이 비어있으면 바로 종성으로
                if self.jong == "":
                    self.jong = c
                    return "", self.composing_text()

                # 종성이 이미 있으면 복합받침 결합 시도
                key = (self.jong, c)
                if key in JONG_COMBINE:
                    self.jong = JONG_COMBINE[key]
                    return "", self.composing_text()

                # 결합 불가: 이전 글자 확정 후 새 초성 시작
                commit += self.composing_text()
                self.reset()
                self.cho = c
                return commit, self.composing_text()

        return "", self.composing_text()

    def backspace(self):
        """
        조합중이면 조합 상태를 한 단계 되돌린다.
        반환: (commit_remove_last_char: bool, new_composing_string)
        - commit_remove_last_char=True면 QLineEdit에서 실제 문자 하나 삭제 필요
        """
        if not self.has_composing():
            return True, ""  # 조합 없으면 실제 문자 삭제

        # 조합 상태 되돌리기: 종 -> 중 -> 초
        if self.jong:
            # 복합받침이면 분리해서 첫번째만 남김
            if self.jong in JONG_SPLIT:
                a, b = JONG_SPLIT[self.jong]
                self.jong = a
            else:
                self.jong = ""
            return False, self.composing_text()

        if self.jung:
            # 복합모음이면 분리해서 첫번째만 남김
            if self.jung in JUNG_SPLIT:
                a, b = JUNG_SPLIT[self.jung]
                self.jung = a
            else:
                self.jung = ""
            return False, self.composing_text()

        if self.cho:
            self.cho = ""
            return False, ""

        return True, ""


class VirtualKeyboard(QWidget):
    """
    mode:
      - "num" : 숫자 키패드(오차/게인/노출/번호검색 등)
      - "full": 전체 키보드(비교데이터 이름 입력 등) + 한/영 토글(자체 조합)
    """
    requestHide = Signal()
    requestReposition = Signal()
    

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)

        self.target: QLineEdit | None = None
        self.mode = "num"
        self.lang = "KO"   # "KO" or "EN"
        self.shift = False

        self.composer = HangulComposer()
        self._composing_len = 0  # QLineEdit에 덮어쓴 조합 문자열 길이

        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(8, 8, 8, 8)
        self.root.setSpacing(6)

        self.title = QLabel("Virtual Keyboard")
        self.title.setStyleSheet("color:white; font-size:14px;")
        self.root.addWidget(self.title)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#999;")
        self.root.addWidget(line)

        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        self.root.addLayout(self.grid)

        self.setStyleSheet("""
            QWidget { background-color:#2b2b2b; border:1px solid #555; }
            QPushButton {
                background-color:#3f4c55; color:white;
                padding:10px; font-size:16px; border-radius:6px;
            }
            QPushButton:pressed { background-color:#0e5a7a; }
        """)

        self._rebuild()

    def attach(self, target: QLineEdit, mode: str):
        self.target = target
        self.mode = mode
        self.shift = False
        # 이름 입력은 한글 가능성이 높아서 KO 기본
        self.lang = "KO" if mode == "full" else "EN"
        self.composer.reset()
        self._composing_len = 0
        self._rebuild()

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _btn(self, text: str, handler, r: int, c: int, rs: int = 1, cs: int = 1):
        b = QPushButton(text)
        b.setFocusPolicy(Qt.NoFocus)

        def wrapped(checked=False):
            # 키보드 조작 중 타겟 유지
            if self.target is not None:
                self.target.setFocus(Qt.OtherFocusReason)
            handler()
            self.requestReposition.emit() 

        b.clicked.connect(wrapped)
        self.grid.addWidget(b, r, c, rs, cs)
        return b
    
    def _rebuild(self):
        self._clear_grid()

        if self.mode == "num":
            self.title.setText("숫자 키패드")
            keys = [
                ["7", "8", "9"],
                ["4", "5", "6"],
                ["1", "2", "3"],
                [".", "0", "-"],
            ]
            for r in range(4):
                for c in range(3):
                    k = keys[r][c]
                    self._btn(k, lambda kk=k: self.on_key(kk), r, c)

            self._btn("⌫", self.on_backspace, 0, 3)
            self._btn("Clear", self.on_clear, 1, 3)
            self._btn("Enter", self.on_enter, 2, 3, 2, 1)
            self._btn("닫기", self.on_close, 4, 0, 1, 4)

        else:
            self.title.setText("전체 키보드 (한/영)")
            # 한/영 모드에 따른 키캡
            if self.lang == "EN":
                row1 = list("qwertyuiop")
                row2 = list("asdfghjkl")
                row3 = list("zxcvbnm")
            else:
                # 두벌식 기본 자모(모바일 느낌): 초성/중성 자모를 직접 찍어 조합
                row1 = list("ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ")
                row2 = list("ㅁㄴㅇㄹㅎㅗㅓㅏㅣ")
                row3 = list("ㅋㅌㅊㅍㅠㅜㅡ")

            # Row1
            for i, k in enumerate(row1):
                self._btn(k, lambda kk=k: self.on_key(kk), 0, i)

            # Row2
            for i, k in enumerate(row2):
                self._btn(k, lambda kk=k: self.on_key(kk), 1, i)

            # Row3 + 기능키
            self._btn("Shift", self.on_shift, 2, 0, 1, 2)
            col = 2
            for k in row3:
                self._btn(k, lambda kk=k: self.on_key(kk), 2, col)
                col += 1
            self._btn("⌫", self.on_backspace, 2, col, 1, 2)

            # Row4
            self._btn("한/영", self.on_lang, 3, 0, 1, 2)
            self._btn("Space", lambda: self.on_key(" "), 3, 2, 1, 5)
            self._btn("Enter", self.on_enter, 3, 7, 1, 2)
            self._btn("닫기", self.on_close, 3, 9, 1, 1)

            # Row5 (숫자/기호 최소)
            self._btn("1", lambda: self.on_key("1"), 4, 0)
            self._btn("2", lambda: self.on_key("2"), 4, 1)
            self._btn("3", lambda: self.on_key("3"), 4, 2)
            self._btn("4", lambda: self.on_key("4"), 4, 3)
            self._btn("5", lambda: self.on_key("5"), 4, 4)
            self._btn(".", lambda: self.on_key("."), 4, 5)
            self._btn("_", lambda: self.on_key("_"), 4, 6)
            self._btn("-", lambda: self.on_key("-"), 4, 7)
            self._btn("Clear", self.on_clear, 4, 8, 1, 2)

        self.adjustSize()

    def _ensure_target(self) -> bool:
        return self.target is not None

    def _remove_composing_from_target(self):
        """QLineEdit에 덮어썼던 조합 문자열을 삭제한다."""
        if not self._ensure_target():
            return
        if self._composing_len <= 0:
            return
        # 커서 앞에서 composing_len 만큼 삭제
        t = self.target.text()
        cur = self.target.cursorPosition()
        start = max(0, cur - self._composing_len)
        new = t[:start] + t[cur:]
        self.target.setText(new)
        self.target.setCursorPosition(start)
        self._composing_len = 0

    def _insert_text(self, s: str):
        if not self._ensure_target() or s == "":
            return
        t = self.target.text()
        cur = self.target.cursorPosition()
        new = t[:cur] + s + t[cur:]
        self.target.setText(new)
        self.target.setCursorPosition(cur + len(s))

    def _set_composing(self, s: str):
        """조합중 문자를 QLineEdit에 '덮어쓰기'로 표현"""
        self._remove_composing_from_target()
        if s:
            self._insert_text(s)
            self._composing_len = len(s)

    def _commit(self, s: str):
        """조합중 문자는 지우고, 커밋 문자를 넣는다."""
        self._remove_composing_from_target()
        if s:
            self._insert_text(s)

    def on_key(self, k: str):
        if not self._ensure_target():
            return

        # 숫자 모드는 그대로 삽입
        if self.mode == "num":
            self._insert_text(k)
            return

        # full 모드
        if self.lang == "EN":
            ch = k
            if len(ch) == 1 and ch.isalpha():
                if self.shift:
                    ch = ch.upper()
                else:
                    ch = ch.lower()
            # EN 입력 시 한글 조합은 flush하고 그대로 넣기
            if self.composer.has_composing():
                self._commit(self.composer.composing_text())
                self.composer.reset()
            self._insert_text(ch)
            # 일반 키 입력 후 shift는 자동 해제(모바일 스타일)
            if self.shift:
                self.shift = False
                self._rebuild()
            return

        # KO 입력: 자모 조합
        commit, comp = self.composer.input_char(k)
        if commit:
            self._commit(commit)
        self._set_composing(comp)

    def on_backspace(self):
        if not self._ensure_target():
            return

        if self.mode == "num":
            # 커서 앞 1글자 삭제
            t = self.target.text()
            cur = self.target.cursorPosition()
            if cur <= 0:
                return
            new = t[:cur-1] + t[cur:]
            self.target.setText(new)
            self.target.setCursorPosition(cur-1)
            return

        # full: 한글 조합 상태를 먼저 backspace
        remove_real, comp = self.composer.backspace()
        if remove_real:
            # 조합 문자열이 화면에 있으면 제거 후, 실제 문자 1개 삭제
            if self._composing_len > 0:
                self._remove_composing_from_target()
                return
            t = self.target.text()
            cur = self.target.cursorPosition()
            if cur <= 0:
                return
            new = t[:cur-1] + t[cur:]
            self.target.setText(new)
            self.target.setCursorPosition(cur-1)
        else:
            self._set_composing(comp)

    def on_clear(self):
        if not self._ensure_target():
            return
        self.target.clear()
        self.composer.reset()
        self._composing_len = 0

    def on_enter(self):
        if not self._ensure_target():
            return
        # 조합중이면 확정
        if self.composer.has_composing():
            self._commit(self.composer.composing_text())
            self.composer.reset()
            self._composing_len = 0
        # QLineEdit의 returnPressed를 강제로 emit
        try:
            self.target.returnPressed.emit()
        except Exception:
            pass

    def on_close(self):
        self.requestHide.emit()

    def on_shift(self):
        self.shift = not self.shift
        self._rebuild()

    def on_lang(self):
        # 조합중이면 확정하고 모드 전환
        if self.composer.has_composing():
            self._commit(self.composer.composing_text())
            self.composer.reset()
            self._composing_len = 0
        self.lang = "EN" if self.lang == "KO" else "KO"
        self.shift = False
        self._rebuild()


class VirtualKeyboardManager(QObject):
    """
    전역 키보드 1개만 공유.
    - 창마다 Manager는 여러 개여도, 실제 VirtualKeyboard는 1개만 존재.
    - 포커스가 바뀌면 owner만 교체해서 "키보드 2개 뜨는 현상" 제거.
    """
    _GLOBAL_KB: VirtualKeyboard | None = None
    _GLOBAL_OWNER: Union["VirtualKeyboardManager", None] = None

    def __init__(self, parent_window: QWidget):
        super().__init__(parent_window)
        self.parent_window = parent_window

        # ✅ 전역 키보드 1개만 만들기
        if VirtualKeyboardManager._GLOBAL_KB is None:
            VirtualKeyboardManager._GLOBAL_KB = VirtualKeyboard(parent_window)
            # requestHide는 "현재 owner"의 hide를 호출하게 연결 (1회만)
            VirtualKeyboardManager._GLOBAL_KB.requestHide.connect(
                lambda: (VirtualKeyboardManager._GLOBAL_OWNER.hide()
                         if VirtualKeyboardManager._GLOBAL_OWNER else None)
            )

        self.kb = VirtualKeyboardManager._GLOBAL_KB

        self._targets: dict[QLineEdit, str] = {}
        self._hide_timer = QTimer()
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

        self._suppress_until_ms = 0
        self._suppress_target = None

    def register(self, line_edit: QLineEdit, mode: str):
        self._targets[line_edit] = mode
        line_edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        if isinstance(obj, QLineEdit):
            if event.type() == QEvent.FocusIn:
                now_ms = int(time.time() * 1000)

                if (self._suppress_target is obj) and (now_ms < self._suppress_until_ms):
                    return False

                self._suppress_target = None
                self._suppress_until_ms = 0

                mode = self._targets.get(obj, "num")
                self.show_for(obj, mode)
                return False

            if event.type() == QEvent.FocusOut:
                # 터치에서 오판 많음 → 자동 숨김 비활성화
                return False

        return super().eventFilter(obj, event)

    def show_for(self, target: QLineEdit, mode: str):
        self._hide_timer.stop()

        # ✅ 기존 owner가 있으면 그 owner의 hide_timer도 멈추고, 키보드 상태를 우리가 가져온다.
        prev = VirtualKeyboardManager._GLOBAL_OWNER
        if prev is not None and prev is not self:
            prev._hide_timer.stop()

        VirtualKeyboardManager._GLOBAL_OWNER = self

        # ✅ 키보드를 현재 창에 “붙이기” (parent_window가 다르면 재부모화)
        #    (네가 찾은 해결: parent_window 붙여야 좌상단 고정 문제가 줄었지)
        if self.kb.parent() is not self.parent_window:
            self.kb.setParent(self.parent_window)
            self.kb.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
            self.kb.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self.kb.attach(target, mode)
        self.kb.adjustSize()
        QApplication.processEvents()  # ✅ 네가 찾은 정답 유지

        kb_w = self.kb.width()
        kb_h = self.kb.height()

        below = target.mapToGlobal(QPoint(0, target.height() + 6))
        above = target.mapToGlobal(QPoint(0, -kb_h - 6))

        screen = QGuiApplication.screenAt(below) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()

        x = below.x()
        y = below.y()
        if y + kb_h > avail.bottom():
            y = above.y()

        # clamp
        if x + kb_w > avail.right():
            x = avail.right() - kb_w
        if x < avail.left():
            x = avail.left()
        if y + kb_h > avail.bottom():
            y = avail.bottom() - kb_h
        if y < avail.top():
            y = avail.top()

        self.kb.move(x, y)
        if not self.kb.isVisible():
            self.kb.show()
        self.kb.raise_()

    def hide(self):
        # 전역 owner가 나 자신일 때만 실제로 숨김
        if VirtualKeyboardManager._GLOBAL_OWNER is self:
            if self.kb.target is not None:
                self._suppress_target = self.kb.target
                self._suppress_until_ms = int(time.time() * 1000) + 150
            self.kb.hide()


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


class CompareSaveDialog(QDialog):
    """비교 데이터 저장(번호/이름 입력 + 목록 표시)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("비교 데이터 저장")
        self.resize(720, 420)

        os.makedirs(COMPARE_DIR, exist_ok=True)

        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        # 상단: 저장되어 있는 목록 + 번호로 검색
        top_row = QHBoxLayout()
        lbl = QLabel("저장되어 있는 데이터 목록...")
        lbl.setStyleSheet("color:white;")
        top_row.addWidget(lbl)

        top_row.addStretch(1)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("번호로 검색")
        self.edit_search.setFixedWidth(220)
        top_row.addWidget(self.edit_search)
        main.addLayout(top_row)

        self.edit_search.setFocusPolicy(Qt.ClickFocus)

        self.list_widget = QListWidget()
        main.addWidget(self.list_widget)

        # 하단 입력
        form = QHBoxLayout()
        lbl_num = QLabel("번호")
        lbl_num.setStyleSheet("color:white;")
        form.addWidget(lbl_num)
        self.edit_number = QLineEdit()
        self.edit_number.setFixedWidth(140)
        form.addWidget(self.edit_number)

        lbl_name = QLabel("이름")
        lbl_name.setStyleSheet("color:white;")
        form.addWidget(lbl_name)
        self.edit_name = QLineEdit()
        self.edit_name.setFixedWidth(260)
        form.addWidget(self.edit_name)

        form.addStretch(1)

        self.btn_save = QPushButton("저장")
        self.btn_save.setFixedSize(140, 48)
        self.btn_save.setStyleSheet("QPushButton { background-color:#4CAF50; color:white; font-size:16px; }")
        form.addWidget(self.btn_save)
        main.addLayout(form)

        # ===== UI 크기/폰트 개선 =====
        self.list_widget.setStyleSheet("font-size:16px;")  # ✅ 목록 글자 키우기

        edit_style = "QLineEdit { font-size:16px; padding:6px; }"
        self.edit_search.setStyleSheet(edit_style)
        self.edit_number.setStyleSheet(edit_style)
        self.edit_name.setStyleSheet(edit_style)

        self.edit_search.setFixedHeight(36)
        self.edit_number.setFixedHeight(40)  # ✅ 입력칸 높이 키우기
        self.edit_name.setFixedHeight(40)

        self.edit_number.setFixedWidth(180)  # ✅ 입력칸 폭 키우기
        self.edit_name.setFixedWidth(360)

        # ✅ 엔터키가 저장(디폴트 버튼)으로 동작하는 것 차단
        self.btn_save.setAutoDefault(False)
        self.btn_save.setDefault(False)

        self.lbl_hint = QLabel("번호와 이름을 입력하면  번호_제품명  폴더에 저장됩니다...")
        self.lbl_hint.setStyleSheet("color:white;")
        main.addWidget(self.lbl_hint)

        self.lbl_err = QLabel("")
        self.lbl_err.setStyleSheet("color:red; font-weight:bold;")
        main.addWidget(self.lbl_err)

        # 이벤트
        self.btn_save.clicked.connect(self.on_save_clicked)
        self.edit_search.returnPressed.connect(self.on_search)
        self.edit_number.textChanged.connect(self.on_number_changed)
        self.list_widget.itemClicked.connect(self.on_item_clicked_fill_form)

        self._populate_list()

        self.selected_folder_name: str | None = None
        self.final_folder_name: str | None = None
        self.overwrite_confirmed: bool = False

        # 배경
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(70, 70, 70))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

        # ============================================================
        # Virtual Keyboard (Dialog)
        # - 번호/검색: num
        # - 이름: full
        # ============================================================
        self.vkb = VirtualKeyboardManager(self)
        self.vkb.register(self.edit_search, "num")
        self.vkb.register(self.edit_number, "num")
        self.vkb.register(self.edit_name, "full")  # ✅ 이름은 전체 키보드

        QTimer.singleShot(0, lambda: self.list_widget.setFocus())


    def _populate_list(self):
        self.list_widget.clear()
        for f in list_compare_folders():
            self.list_widget.addItem(f)

    def on_search(self):
        num = self.edit_search.text().strip()
        f = find_folder_by_number(num)
        if f is None:
            self.lbl_err.setText("해당 번호로 시작하는 폴더를 찾지 못했습니다.")
            return
        self.lbl_err.setText("")
        # 목록에서 선택
        items = self.list_widget.findItems(f, Qt.MatchExactly)
        if items:
            self.list_widget.setCurrentItem(items[0])

    def on_number_changed(self):
        num = self.edit_number.text().strip()
        if num == "":
            self.lbl_hint.setText("번호와 이름을 입력하면  번호_제품명  폴더에 저장됩니다...")
            return
        f = find_folder_by_number(num)
        if f:
            self.lbl_hint.setText(f"⚠ 지정된 번호({num})에 데이터가 존재합니다: {f}")
        else:
            self.lbl_hint.setText("번호와 이름을 입력하면  번호_제품명  폴더에 저장됩니다...")

    def on_save_clicked(self):
        num = sanitize_folder_component(self.edit_number.text())
        name = sanitize_folder_component(self.edit_name.text())

        if num == "" or name == "":
            self.lbl_err.setText("번호 또는 이름이 지정되지 않았습니다.")
            return

        folder_name = f"{num}_{name}"
        folder_path = os.path.join(COMPARE_DIR, folder_name)

        # 중복 번호 처리: 같은 번호_* 폴더가 이미 있으면 경고 + 덮어쓰기 여부
        existing = find_folder_by_number(num)
        if existing is not None and existing != folder_name:
            # 번호는 같지만 이름이 다를 수 있음 → 번호 충돌로 봄
            reply = QMessageBox.question(
                self,
                "경고!",
                "지정된 번호에 데이터가 존재합니다!\n덮어쓰시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                self.lbl_err.setText("저장이 취소되었습니다.")
                return
            # 기존 번호 폴더 삭제 후 새 폴더로 저장
            try:
                shutil.rmtree(os.path.join(COMPARE_DIR, existing))
            except Exception:
                pass

        elif os.path.isdir(folder_path):
            # 정확히 같은 폴더가 존재 → 덮어쓰기 질문
            reply = QMessageBox.question(
                self,
                "경고!",
                "지정된 번호에 데이터가 존재합니다!\n덮어쓰시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                self.lbl_err.setText("저장이 취소되었습니다.")
                return
            try:
                shutil.rmtree(folder_path)
            except Exception:
                pass

        self.final_folder_name = folder_name
        self.overwrite_confirmed = True
        self.accept()

    def on_item_clicked_fill_form(self, item):
        """저장된 폴더 클릭 시 번호/이름 입력칸 자동 채움."""
        if item is None:
            return
        text = (item.text() or "").strip()
        if text.startswith("("):
            return

        # 폴더명: "번호_이름" 형태 가정
        if "_" in text:
            num, name = text.split("_", 1)
        else:
            num, name = text, ""

        self.edit_number.setText(num)
        self.edit_name.setText(name)
        self.lbl_err.setText("")


class CompareLoadDialog(QDialog):
    """비교 데이터 불러오기(목록 표시 + 번호 검색)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("비교 데이터 불러오기")
        self.resize(720, 420)

        os.makedirs(COMPARE_DIR, exist_ok=True)

        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        # 상단: 저장되어 있는 목록 + 번호로 검색
        top_row = QHBoxLayout()
        lbl = QLabel("저장되어 있는 데이터 목록...")
        lbl.setStyleSheet("color:white;")
        top_row.addWidget(lbl)

        top_row.addStretch(1)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("번호로 검색")
        self.edit_search.setFixedWidth(220)
        top_row.addWidget(self.edit_search)
        main.addLayout(top_row)

        self.edit_search.setFocusPolicy(Qt.ClickFocus)

        self.list_widget = QListWidget()
        main.addWidget(self.list_widget)

        bottom = QHBoxLayout()
        self.lbl_hint = QLabel("데이터를 지정해 주세요.")
        self.lbl_hint.setStyleSheet("color:white;")
        bottom.addWidget(self.lbl_hint)

        bottom.addStretch(1)

        self.btn_load = QPushButton("불러오기")
        self.btn_load.setFixedSize(140, 48)
        self.btn_load.setStyleSheet("QPushButton { background-color:#4CAF50; color:white; font-size:16px; }")
        bottom.addWidget(self.btn_load)

        # ===== UI 크기/폰트 개선 =====
        self.list_widget.setStyleSheet("font-size:16px;")  # ✅ 목록 글자 키우기
        self.edit_search.setStyleSheet("QLineEdit { font-size:16px; padding:6px; }")
        self.edit_search.setFixedHeight(36)

        # ✅ 엔터키가 불러오기(디폴트 버튼)로 동작하는 것 차단
        self.btn_load.setAutoDefault(False)
        self.btn_load.setDefault(False)

        # ✅ 더블클릭으로 불러오기
        self.list_widget.itemDoubleClicked.connect(lambda *_: self.on_load_clicked())

        main.addLayout(bottom)

        self.lbl_err = QLabel("")
        self.lbl_err.setStyleSheet("color:red; font-weight:bold;")
        main.addWidget(self.lbl_err)

        self.btn_load.clicked.connect(self.on_load_clicked)
        self.edit_search.returnPressed.connect(self.on_search)

        self._populate_list()

        self.selected_folder: str | None = None

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(70, 70, 70))
        self.setAutoFillBackground(True)
        self.setPalette(pal)

        # ============================================================
        # Virtual Keyboard (Dialog)
        # - 검색: num
        # ============================================================
        self.vkb = VirtualKeyboardManager(self)
        self.vkb.register(self.edit_search, "num")

        QTimer.singleShot(0, lambda: self.list_widget.setFocus())


    def _populate_list(self):
        self.list_widget.clear()
        folders = list_compare_folders()
        if not folders:
            self.list_widget.addItem("(저장된 비교 데이터가 없습니다)")
            self.list_widget.setEnabled(False)
            self.btn_load.setEnabled(False)
        else:
            for f in folders:
                self.list_widget.addItem(f)

    def on_search(self):
        num = self.edit_search.text().strip()
        f = find_folder_by_number(num)
        if f is None:
            self.lbl_err.setText("해당 번호로 시작하는 폴더를 찾지 못했습니다.")
            return
        self.lbl_err.setText("")
        items = self.list_widget.findItems(f, Qt.MatchExactly)
        if items:
            self.list_widget.setCurrentItem(items[0])

    def on_load_clicked(self):
        if not self.list_widget.isEnabled():
            self.lbl_err.setText("불러올 데이터가 없습니다.")
            return
        row = self.list_widget.currentRow()
        if row < 0:
            self.lbl_err.setText("데이터를 지정해 주세요.")
            return
        folder = self.list_widget.currentItem().text().strip()
        if folder.startswith("("):
            self.lbl_err.setText("데이터를 지정해 주세요.")
            return
        self.selected_folder = folder
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

        # comparedata 폴더 자동 생성
        os.makedirs(COMPARE_DIR, exist_ok=True)

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
        # 최근 2D 컬러/그레이 이미지
        self.last_color_qimage: QImage | None = None

        # 비교 기준 데이터 (좌측)
        self.compare_pointcloud = None
        self.compare_colors = None
        self.compare_depth_qimage: QImage | None = None
        self.compare_depth_array: np.ndarray | None = None
        # 비교용 2D 이미지
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
        # 상단: Point / Depths / Image + Save Recent Data + (Save/Load Compare Data) + 시간 + Exit
        # =======================
        top_wrap = QVBoxLayout()
        top_wrap.setSpacing(6)

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

        # 버튼 순서/매핑: Point, Depths, Image
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

        # Save Recent Data 버튼
        self.btn_save_files = QPushButton("Save Recent Data")
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

        # 상단 우측: 로그 초기화 버튼 (기존 Exit 자리 활용)
        self.btn_refresh_top = QPushButton("로그 초기화")
        self.btn_refresh_top.setStyleSheet(
            "QPushButton { background-color:white; color:black; padding:4px; font-size:12px; }"
        )
        self.btn_refresh_top.setFixedWidth(90)
        self.btn_refresh_top.setFixedHeight(32)
        self.btn_refresh_top.clicked.connect(self.on_refresh_clicked)
        top_layout.addWidget(self.btn_refresh_top)

        top_wrap.addLayout(top_layout)

        # ★ 추가: Save Compare Data / Load Compare Data (상단에 배치)
        top2_layout = QHBoxLayout()
        top2_layout.setSpacing(8)

        self.btn_save_compare_data = QPushButton("Save Compare Data")
        self.btn_save_compare_data.setStyleSheet(
            "QPushButton { background-color: #0e5a7a; color: white; padding: 6px; font-size: 14px; }"
            "QPushButton:disabled { background-color: #4a6c7a; }"
        )
        self.btn_save_compare_data.clicked.connect(self.on_save_compare_data_clicked)
        top2_layout.addWidget(self.btn_save_compare_data)

        self.btn_load_compare_data = QPushButton("Load Compare Data")
        self.btn_load_compare_data.setStyleSheet(
            "QPushButton { background-color: #0e5a7a; color: white; padding: 6px; font-size: 14px; }"
            "QPushButton:disabled { background-color: #4a6c7a; }"
        )
        self.btn_load_compare_data.clicked.connect(self.on_load_compare_data_clicked)
        top2_layout.addWidget(self.btn_load_compare_data)

        self.btn_reset_compare_data = QPushButton("Reset Compare Data")
        self.btn_reset_compare_data.setStyleSheet(
            "QPushButton { background-color: #0e5a7a; color: white; padding: 6px; font-size: 14px; }"
        )
        self.btn_reset_compare_data.clicked.connect(self.on_reset_compare_data_clicked)
        top2_layout.addWidget(self.btn_reset_compare_data)

        # Upload Git 버튼
        self.btn_upload_git = QPushButton("Upload Git")
        self.btn_upload_git.setStyleSheet(btn_style)
        self.btn_upload_git.clicked.connect(self.on_upload_git_clicked)
        top2_layout.addWidget(self.btn_upload_git)


        top2_layout.addStretch(1)
        top_wrap.addLayout(top2_layout)

        main_layout.addLayout(top_wrap)

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

        self.edit_tol_distance = QLineEdit("10")     # 기본값 10mm
        self.edit_tol_distance.setAlignment(Qt.AlignCenter)
        self.edit_tol_distance.setMaximumWidth(100)
        right_side.addWidget(self.edit_tol_distance)

        lbl_ratio = QLabel("오차 비율(%)")
        lbl_ratio.setAlignment(Qt.AlignCenter)
        lbl_ratio.setStyleSheet("color:white; font-size:12px;")
        right_side.addWidget(lbl_ratio)

        self.edit_tol_ratio = QLineEdit("2")         # 기본값 2%
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

        lbl_cam_param = QLabel("카메라 파라미터")
        lbl_cam_param.setAlignment(Qt.AlignCenter)
        lbl_cam_param.setStyleSheet("color:white; font-size:12px; font-weight:bold;")
        right_side.addWidget(lbl_cam_param)

        lbl_gain = QLabel("게인(0~16 dB)")
        lbl_gain.setAlignment(Qt.AlignCenter)
        lbl_gain.setStyleSheet("color:white; font-size:12px;")
        right_side.addWidget(lbl_gain)

        self.edit_gain_db = QLineEdit("12")   # ✅ 기본값 12 dB
        self.edit_gain_db.setAlignment(Qt.AlignCenter)
        self.edit_gain_db.setMaximumWidth(100)
        right_side.addWidget(self.edit_gain_db)

        lbl_exp = QLabel("노출 시간(ms)")
        lbl_exp.setAlignment(Qt.AlignCenter)
        lbl_exp.setStyleSheet("color:white; font-size:12px;")
        right_side.addWidget(lbl_exp)

        self.edit_exposure_ms = QLineEdit("6.0")  # ✅ 기본값 6.0 ms
        self.edit_exposure_ms.setAlignment(Qt.AlignCenter)
        self.edit_exposure_ms.setMaximumWidth(100)
        right_side.addWidget(self.edit_exposure_ms)

        lbl_note = QLabel("2D, 3D 동일 값 적용")
        lbl_note.setAlignment(Qt.AlignCenter)
        lbl_note.setStyleSheet("color:white; font-size:11px;")
        right_side.addWidget(lbl_note)

        self.btn_apply_cam = QPushButton("apply")
        self.btn_apply_cam.setStyleSheet(
            "QPushButton { background-color:#0e5a7a; color:white; padding:6px; font-size:14px; }"
        )
        self.btn_apply_cam.setFixedWidth(100)
        self.btn_apply_cam.setFixedHeight(32)
        self.btn_apply_cam.clicked.connect(self.on_apply_camera_params_clicked)
        right_side.addWidget(self.btn_apply_cam)

        right_side.addStretch(1)

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

        # ★ 기존 "Save Compare Data" → "Define Compare Data"
        self.btn_define_compare = QPushButton("Define\nCompare\nData")
        self.btn_define_compare.setStyleSheet(bottom_btn_style)
        self.btn_define_compare.clicked.connect(self.on_define_compare_clicked)
        btn_col.addWidget(self.btn_define_compare)

        bottom_layout.addLayout(btn_col, 1)

        main_layout.addLayout(bottom_layout, 1)

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(70, 70, 70))
        self.setPalette(pal)

        # ============================================================
        # Virtual Keyboard 적용
        # ============================================================
        self.vkb = VirtualKeyboardManager(self)

        # 숫자키패드 대상들
        self.vkb.register(self.edit_tol_distance, "num")
        self.vkb.register(self.edit_tol_ratio, "num")
        self.vkb.register(self.edit_gain_db, "num")
        self.vkb.register(self.edit_exposure_ms, "num")


        if pg is None or GLViewWidget is None:
            self.append_log("[ERROR] pyqtgraph / PyOpenGL 미설치. 3D 뷰어를 사용하려면 pip install pyqtgraph PyOpenGL")

    # ===== 로그 도우미 =====
    def append_log(self, text: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{now}] {text}")

    # ===== 모드 전환 =====
    def on_mode_changed(self):
        if self.btn_point.isChecked():
            if self.compare_pointcloud is not None:
                self.update_compare_pointcloud_view()

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

    def on_upload_git_clicked(self):
        script_path = os.path.join(BASE_DIR, "UploadGit.py")

        if not os.path.isfile(script_path):
            self.append_log(f"[ERROR] UploadGit.py 를 찾을 수 없습니다: {script_path}")
            return

        self.append_log("[GIT] UploadGit.py 실행...")

        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                [sys.executable, "-X", "utf8", script_path],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()

            if out:
                for line in out.splitlines():
                    self.append_log(f"[UploadGit.py] {line}")

            if err:
                for line in err.splitlines():
                    self.append_log(f"[UploadGit.py][ERR] {line}")

            if result.returncode == 0:
                self.append_log("[GIT] UploadGit.py 완료 (returncode=0)")
            else:
                self.append_log(f"[GIT] UploadGit.py 실패 (returncode={result.returncode})")

        except Exception as e:
            self.append_log(f"[ERROR] UploadGit.py 실행 중 예외: {e}")

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

    # ===== 최근 스캔 데이터 파일로 저장 =====
    def on_save_files_clicked(self):
        """
        result/날짜시간/ 폴더에
        - PNG : 2D 카메라 이미지
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

        # PNG
        if self.last_color_qimage is not None:
            png_path = os.path.join(save_dir, f"{base}.png")
            ok = self.last_color_qimage.save(png_path, "PNG")
            if ok:
                self.append_log(f"PNG(2D 이미지) 저장 완료: {png_path}")
            else:
                self.append_log("[ERROR] PNG 저장 실패")
        else:
            self.append_log("[WARN] PNG 저장 생략: 2D 이미지 없음")

        # TIFF
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

        # PLY
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
                                r, g, b = [int(np.clip(v, 0, 255)) for v in c]
                                f.write(f"{x} {y} {z} {r} {g} {b}\n")
                            else:
                                f.write(f"{x} {y} {z}\n")

                self.append_log(f"PLY 저장 완료: {ply_path}")
            except Exception as e:
                self.append_log(f"[ERROR] PLY 저장 중 예외: {e}")
        else:
            self.append_log("[WARN] PLY 저장 생략: 포인트 클라우드 없음")

        self.append_log(f"스캔 데이터 저장 완료: {save_dir}")

    # ===== 기존: 비교데이터 "정의" (last_* → compare_*) =====
    def on_define_compare_clicked(self):
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

        self.compare_color_qimage = None if self.last_color_qimage is None else self.last_color_qimage.copy()

        if self.btn_point.isChecked():
            self.update_compare_pointcloud_view()
        elif self.btn_depth.isChecked():
            self.update_compare_depth_view()
        elif self.btn_image.isChecked():
            self.update_compare_image_view()

        self.append_log("Define Compare Data 완료 (현재 스캔을 비교 기준으로 설정).")
        self.update_tolerance_display()

    # ===== NEW: define된 비교데이터(compare_*)를 별도 폴더로 저장 =====
    def on_save_compare_data_clicked(self):
        if (
            self.compare_pointcloud is None
            and self.compare_depth_array is None
            and self.compare_color_qimage is None
        ):
            self.append_log("[ERROR] 저장할 비교 데이터가 없습니다. 먼저 Define Compare Data를 수행하세요.")
            return

        dlg = CompareSaveDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.append_log("[INFO] 비교 데이터 저장 취소.")
            return

        folder_name = dlg.final_folder_name
        if not folder_name:
            self.append_log("[ERROR] 폴더명이 생성되지 않아 저장을 중단합니다.")
            return

        try:
            self.save_compare_bundle(folder_name)
            self.append_log(f"[INFO] 비교 데이터 저장 완료: {os.path.join(COMPARE_DIR, folder_name)}")
        except Exception as e:
            self.append_log(f"[ERROR] 비교 데이터 저장 실패: {e}")

    # ===== NEW: 저장된 비교데이터 불러오기 =====
    def on_load_compare_data_clicked(self):
        dlg = CompareLoadDialog(self)
        if dlg.exec() != QDialog.Accepted:
            self.append_log("[INFO] 비교 데이터 불러오기 취소.")
            return
        folder = dlg.selected_folder
        if not folder:
            self.append_log("[ERROR] 불러올 폴더가 선택되지 않았습니다.")
            return

        try:
            self.load_compare_bundle(folder)
            self.append_log(f"[INFO] 비교 데이터 불러오기 완료: {os.path.join(COMPARE_DIR, folder)}")

            # 화면 갱신
            if self.btn_point.isChecked():
                self.update_compare_pointcloud_view()
                if self.show_diff_overlay and self.last_pointcloud is not None:
                    self.update_recent_pointcloud_diff_view()
            elif self.btn_depth.isChecked():
                self.update_compare_depth_view()
                if self.last_depth_qimage is not None:
                    self.update_recent_depth_view()
            elif self.btn_image.isChecked():
                self.update_compare_image_view()
                if self.last_color_qimage is not None:
                    self.update_recent_image_view()

            self.update_tolerance_display()

        except Exception as e:
            self.append_log(f"[ERROR] 비교 데이터 불러오기 실패: {e}")

    def save_compare_bundle(self, folder_name: str):
        """comparedata\\folder_name\\folder_name.(png/tiff/ply)로 저장."""
        os.makedirs(COMPARE_DIR, exist_ok=True)

        folder_name = sanitize_folder_component(folder_name)
        save_dir = os.path.join(COMPARE_DIR, folder_name)

        # 덮어쓰기는 다이얼로그에서 이미 결정됨 → 여기서는 그냥 생성
        os.makedirs(save_dir, exist_ok=True)

        base = folder_name

        # --- NEW: 허용오차 설정도 함께 저장(meta.json) ---
        meta_path = os.path.join(save_dir, "meta.json")
        meta = {
            "tol_distance_mm": float(self.get_threshold_mm()),
            "tol_ratio_percent": float(self.get_allow_ratio() * 100.0),
            "gain_db": float(self._get_gain_db()),
            "exposure_ms": float(self._get_exposure_ms()),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bundle_name": base,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.append_log(f"[COMPARE] META 저장: {meta_path}")

        # PNG
        if self.compare_color_qimage is not None:
            png_path = os.path.join(save_dir, f"{base}.png")
            ok = self.compare_color_qimage.save(png_path, "PNG")
            if ok:
                self.append_log(f"[COMPARE] PNG 저장: {png_path}")
            else:
                raise RuntimeError("COMPARE PNG 저장 실패")
        else:
            self.append_log("[COMPARE] PNG 저장 생략: 2D 이미지 없음")

        # TIFF
        if self.compare_depth_array is not None:
            tiff_path = os.path.join(save_dir, f"{base}.tiff")
            saved = False
            depth = self.compare_depth_array.astype(np.float32)
            if Image is not None:
                img = Image.fromarray(depth)
                img.save(tiff_path, format="TIFF")
                saved = True
            elif cv2 is not None:
                cv2.imwrite(tiff_path, depth)
                saved = True

            if saved:
                self.append_log(f"[COMPARE] TIFF 저장: {tiff_path}")
            else:
                raise RuntimeError("COMPARE TIFF 저장 실패 (PIL/OpenCV 없음)")
        else:
            self.append_log("[COMPARE] TIFF 저장 생략: depth 없음")

        # PLY
        if self.compare_pointcloud is not None and self.compare_pointcloud.size > 0:
            ply_path = os.path.join(save_dir, f"{base}.ply")
            pts = self.compare_pointcloud
            cols = self.compare_colors

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
                            r, g, b = [int(np.clip(v, 0, 255)) for v in c]
                            f.write(f"{x} {y} {z} {r} {g} {b}\n")
                        else:
                            f.write(f"{x} {y} {z}\n")
            self.append_log(f"[COMPARE] PLY 저장: {ply_path}")
        else:
            self.append_log("[COMPARE] PLY 저장 생략: pointcloud 없음")

    def load_compare_bundle(self, folder_name: str):
        """comparedata\\folder_name\\folder_name.(png/tiff/ply)에서 로드."""
        folder_name = sanitize_folder_component(folder_name)
        load_dir = os.path.join(COMPARE_DIR, folder_name)
        if not os.path.isdir(load_dir):
            raise FileNotFoundError(f"폴더가 없습니다: {load_dir}")

        base = folder_name
        meta_path = os.path.join(load_dir, "meta.json")
        png_path = os.path.join(load_dir, f"{base}.png")
        tiff_path = os.path.join(load_dir, f"{base}.tiff")
        ply_path = os.path.join(load_dir, f"{base}.ply")

        # PNG → compare_color_qimage
        self.compare_color_qimage = None
        if os.path.isfile(png_path):
            qimg = None
            if Image is not None:
                try:
                    img = Image.open(png_path).convert("RGB")
                    arr = np.array(img)
                    h, w, _ = arr.shape
                    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                except Exception:
                    qimg = None
            if qimg is None and cv2 is not None:
                img = cv2.imread(png_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, _ = img.shape
                    qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.compare_color_qimage = qimg

        # TIFF → compare_depth_array / compare_depth_qimage(컬러맵)
        self.compare_depth_array = None
        self.compare_depth_qimage = None
        if os.path.isfile(tiff_path):
            depth_np = None
            if cv2 is not None:
                d = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
                if d is not None:
                    depth_np = d.astype(np.float32)
            if depth_np is None and Image is not None:
                img = Image.open(tiff_path)
                depth_np = np.array(img).astype(np.float32)

            if depth_np is not None:
                self.compare_depth_array = depth_np
                rgb_uint8 = depth_to_color_image(depth_np)
                rgb_uint8 = np.ascontiguousarray(rgb_uint8)
                h, w, _ = rgb_uint8.shape
                self.compare_depth_qimage = QImage(
                    rgb_uint8.data, w, h, 3 * w, QImage.Format_RGB888
                ).copy()

        # PLY → compare_pointcloud / compare_colors
        self.compare_pointcloud = None
        self.compare_colors = None
        if os.path.isfile(ply_path):
            if o3d is not None:
                pcd = o3d.io.read_point_cloud(ply_path)
                pts = np.asarray(pcd.points)
                if pts.ndim == 2 and pts.shape[1] == 3:
                    mask = np.isfinite(pts).all(axis=1)
                    pts = pts[mask]
                    self.compare_pointcloud = pts.astype(np.float32)
                    if pcd.has_colors():
                        cols = np.asarray(pcd.colors)
                        cols = cols[mask]
                        self.compare_colors = cols.astype(np.float32)
                    else:
                        self.compare_colors = None
            else:
                # open3d 없으면 최소: point만 읽는 간단 파서(ASCII PLY)
                pts = []
                cols = []
                with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
                    header = True
                    has_color = False
                    for line in f:
                        line = line.strip()
                        if header:
                            if line.startswith("property uchar red"):
                                has_color = True
                            if line == "end_header":
                                header = False
                            continue
                        parts = line.split()
                        if len(parts) >= 3:
                            x, y, z = map(float, parts[:3])
                            pts.append([x, y, z])
                            if has_color and len(parts) >= 6:
                                r, g, b = map(int, parts[3:6])
                                cols.append([r / 255.0, g / 255.0, b / 255.0])
                if pts:
                    self.compare_pointcloud = np.array(pts, dtype=np.float32)
                    self.compare_colors = np.array(cols, dtype=np.float32) if cols else None

        # --- NEW: 허용오차 설정 불러오기(meta.json) ---
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                dist_mm = meta.get("tol_distance_mm", None)
                ratio_pct = meta.get("tol_ratio_percent", None)
                gain_db = meta.get("gain_db", None)
                exposure_ms = meta.get("exposure_ms", None)

                if dist_mm is not None:
                    self.edit_tol_distance.setText(str(dist_mm))
                if ratio_pct is not None:
                    self.edit_tol_ratio.setText(str(ratio_pct))

                # ✅ 게인/노출 입력칸 자동 기입
                if gain_db is not None:
                    self.edit_gain_db.setText(str(gain_db))
                if exposure_ms is not None:
                    self.edit_exposure_ms.setText(str(exposure_ms))

                self.append_log("[COMPARE] META 로드 → 허용오차/카메라 파라미터 자동 Apply")

                # ✅ 허용오차 Apply 자동 실행
                self.on_apply_tolerance_clicked()

                # ✅ 카메라 파라미터 Apply 자동 실행 (실카메라면 적용, 샘플모드면 스킵 로그)
                self.on_apply_camera_params_clicked()

            except Exception as e:
                self.append_log(f"[WARN] meta.json 로드 실패: {e}")
        else:
            self.append_log("[INFO] meta.json 없음(현재 설정 유지)")

    def on_reset_compare_data_clicked(self):
        # 정의된 비교 데이터(좌측)를 완전히 비움
        self.compare_pointcloud = None
        self.compare_colors = None
        self.compare_depth_qimage = None
        self.compare_depth_array = None
        self.compare_color_qimage = None

        self.append_log("[INFO] 비교 데이터(Define Compare Data)가 리셋되었습니다.")

        # 좌측 뷰어 화면 갱신
        if self.btn_point.isChecked():
            self.compare_viewer.label.setPixmap(QPixmap())
            self.compare_viewer.label.setText("스캔 데이터 출력")
            self.compare_viewer.label.setAlignment(Qt.AlignCenter)
            self.compare_viewer.label.show()
            if self.compare_viewer.gl_view is not None:
                self.compare_viewer.gl_view.hide()

            # 우측은 diff가 켜져있어도 비교 기준이 없으니 일반 뷰로
            if self.last_pointcloud is not None:
                self.update_recent_pointcloud_view()

        elif self.btn_depth.isChecked():
            self.compare_viewer.label.setPixmap(QPixmap())
            self.compare_viewer.label.setText("스캔 데이터 출력")
            self.compare_viewer.label.setAlignment(Qt.AlignCenter)
            self.compare_viewer.label.show()
            if self.last_depth_qimage is not None:
                self.update_recent_depth_view()

        elif self.btn_image.isChecked():
            self.compare_viewer.label.setPixmap(QPixmap())
            self.compare_viewer.label.setText("스캔 데이터 출력")
            self.compare_viewer.label.setAlignment(Qt.AlignCenter)
            self.compare_viewer.label.show()
            if self.last_color_qimage is not None:
                self.update_recent_image_view()

        # 허용오차 표시도 비교 기준이 없으니 '-'로 내려가게
        self.update_tolerance_display()

    # ===== 실제 캡쳐 동작 =====
    def capture_both_depth_and_point(self):
        """
        2D 이미지 + Depth map + Point cloud 를 한 번에 받아서
        '최근 데이터' 변수들(last_*)에 저장.
        """
        if USE_REAL_CAMERA:
            frame2d_and_3d = Frame2DAnd3D()
            show_error(self.camera.capture_2d_and_3d_with_normal(frame2d_and_3d))

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

            # 샘플 2D 이미지
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
        res = self.make_pointcloud_diff_for_view()
        if res is None:
            self.append_log("[WARN] Point diff 뷰 생성 실패 - 일반 포인트 표시로 대체합니다.")
            self.update_recent_pointcloud_view()
            return

        pts_aligned, colors = res
        self._update_viewer_with_pointcloud(
            self.scan_viewer, pts_aligned, colors
        )

    def compute_depth_diff_stats(self,
                                 ref_depth: np.ndarray,
                                 cur_depth: np.ndarray):
        if ref_depth is None or cur_depth is None:
            return None

        d_ref = ref_depth.astype(np.float32)
        d_cur = cur_depth.astype(np.float32)

        h = min(d_ref.shape[0], d_cur.shape[0])
        w = min(d_ref.shape[1], d_cur.shape[1])
        d_ref = d_ref[:h, :w]
        d_cur = d_cur[:h, :w]

        valid_ref = np.isfinite(d_ref) & (d_ref > 0)
        valid_cur = np.isfinite(d_cur) & (d_cur > 0)
        valid_any = valid_ref | valid_cur

        if not np.any(valid_any):
            return None

        both_valid = valid_ref & valid_cur

        diff_mm = np.zeros_like(d_ref, dtype=np.float32)
        diff_mm[both_valid] = np.abs(d_cur[both_valid] - d_ref[both_valid])

        threshold_mm = self.get_threshold_mm()
        only_one_valid = valid_any & (~both_valid)
        diff_mm[only_one_valid] = threshold_mm

        valid_vals = diff_mm[valid_any]
        mean_mm = float(valid_vals.mean())
        max_mm = float(valid_vals.max())

        diff_ratio = float(np.mean(valid_vals >= threshold_mm))

        mean_cm = mean_mm / 10.0
        max_cm = max_mm / 10.0

        return mean_cm, max_cm, diff_ratio, threshold_mm

    def compute_pointcloud_diff_stats(self,
                                      ref_pts: np.ndarray | None,
                                      cur_pts: np.ndarray | None):
        if o3d is None:
            self.append_log("[WARN] open3d 미설치 - 포인트 클라우드 비교를 건너뜁니다.")
            return None

        if ref_pts is None or cur_pts is None:
            return None

        ref = np.asarray(ref_pts, dtype=np.float32)
        cur = np.asarray(cur_pts, dtype=np.float32)
        if ref.size == 0 or cur.size == 0:
            return None

        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(ref)

        pcd_cur = o3d.geometry.PointCloud()
        pcd_cur.points = o3d.utility.Vector3dVector(cur)

        voxel_size = 2.0
        if voxel_size > 0:
            pcd_ref = pcd_ref.voxel_down_sample(voxel_size)
            pcd_cur = pcd_cur.voxel_down_sample(voxel_size)

        if len(pcd_ref.points) == 0 or len(pcd_cur.points) == 0:
            return None

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

        pcd_cur_aligned = pcd_cur.transform(T)

        d_cur_to_ref = np.asarray(
            pcd_cur_aligned.compute_point_cloud_distance(pcd_ref)
        )
        d_ref_to_cur = np.asarray(
            pcd_ref.compute_point_cloud_distance(pcd_cur_aligned)
        )

        if d_cur_to_ref.size == 0 or d_ref_to_cur.size == 0:
            return None

        dists = np.concatenate([d_cur_to_ref, d_ref_to_cur])

        threshold_mm = self.get_threshold_mm()
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

        valid_ref = np.isfinite(d_ref) & (d_ref > 0)
        valid_cur = np.isfinite(d_cur) & (d_cur > 0)

        valid_any = valid_ref | valid_cur
        if not np.any(valid_any):
            self.append_log("[DEBUG] diff: 유효한 depth 픽셀이 없습니다.")
            return None

        both_valid = valid_ref & valid_cur

        diff_mm = np.zeros_like(d_ref, dtype=np.float32)
        diff_mm[both_valid] = np.abs(d_cur[both_valid] - d_ref[both_valid])

        threshold_mm = self.get_threshold_mm()
        only_one_valid = valid_any & (~both_valid)
        diff_mm[only_one_valid] = threshold_mm

        valid_vals = diff_mm[valid_any]
        mean_diff_mm = float(valid_vals.mean())
        max_diff_mm = float(valid_vals.max())

        self.append_log(
            f"[DEBUG] diff 통계: mean={mean_diff_mm:.2f} mm, "
            f"max={max_diff_mm:.2f} mm, thr={threshold_mm:.1f} mm"
        )

        mask_diff = (diff_mm >= threshold_mm) & valid_any

        rgb_uint8 = depth_to_color_image(d_cur)
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
        if o3d is None:
            self.append_log("[WARN] open3d 미설치 - 포인트 diff 뷰를 만들 수 없습니다.")
            return None

        if self.compare_pointcloud is None or self.last_pointcloud is None:
            return None

        ref = np.asarray(self.compare_pointcloud, dtype=np.float32)
        cur = np.asarray(self.last_pointcloud, dtype=np.float32)
        if ref.size == 0 or cur.size == 0:
            return None

        pcd_ref = o3d.geometry.PointCloud()
        pcd_ref.points = o3d.utility.Vector3dVector(ref)

        pcd_cur = o3d.geometry.PointCloud()
        pcd_cur.points = o3d.utility.Vector3dVector(cur)

        voxel_size = 2.0
        if voxel_size > 0:
            pcd_ref = pcd_ref.voxel_down_sample(voxel_size)
            pcd_cur = pcd_cur.voxel_down_sample(voxel_size)

        if len(pcd_ref.points) == 0 or len(pcd_cur.points) == 0:
            return None

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

        pts_cur = np.asarray(pcd_cur_aligned.points)
        pts_ref = np.asarray(pcd_ref.points)
        if pts_cur.size == 0 or pts_ref.size == 0:
            return None

        d_cur_to_ref = np.asarray(
            pcd_cur_aligned.compute_point_cloud_distance(pcd_ref)
        )
        d_ref_to_cur = np.asarray(
            pcd_ref.compute_point_cloud_distance(pcd_cur_aligned)
        )

        threshold_mm = self.get_threshold_mm()

        dist_clipped = np.clip(d_cur_to_ref, 0.0, threshold_mm)
        norm = dist_clipped / (threshold_mm + 1e-6)

        h = (1.0 - norm) * (1.0 / 3.0)  # green ~ red
        s = np.ones_like(h, dtype=np.float32)
        v = np.ones_like(h, dtype=np.float32)

        r, g, b = hsv_to_rgb_np(h, s, v)
        colors_cur = np.stack([r, g, b], axis=1).astype(np.float32)

        mask_large_cur = d_cur_to_ref >= threshold_mm
        colors_cur[mask_large_cur] = 1.0

        mask_large_ref = d_ref_to_cur >= threshold_mm
        pts_ref_far = pts_ref[mask_large_ref]

        if pts_ref_far.size > 0:
            colors_ref_far = np.ones_like(pts_ref_far, dtype=np.float32)
            pts_combined = np.vstack([pts_cur, pts_ref_far])
            colors_combined = np.vstack([colors_cur, colors_ref_far])
        else:
            pts_combined = pts_cur
            colors_combined = colors_cur

        return pts_combined, colors_combined

    def get_threshold_mm(self) -> float:
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
        thr_mm = self.get_threshold_mm()
        ratio = self.get_allow_ratio()

        self.append_log(
            f"[INFO] 허용 오차 기준 재적용: 거리={thr_mm:.1f}mm, 비율={ratio*100:.2f}%"
        )

        self.update_tolerance_display()

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
        depth_result = None
        pc_result = None

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

        result = pc_result if pc_result is not None else depth_result

        if result is None:
            self.label_tol_indicator.setText("-")
            self.label_tol_indicator.setStyleSheet(
                "background-color:white; color:black; font-size:32px; font-weight:bold;"
            )
            return

        mean_cm, max_cm, diff_ratio, threshold_mm = result

        thr_mm = self.get_threshold_mm()
        thr_cm = thr_mm / 10.0
        ratio_limit = self.get_allow_ratio()

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

    def on_diff_toggle_clicked(self):
        self.show_diff_overlay = self.btn_diff_toggle.isChecked()
        if self.show_diff_overlay:
            self.btn_diff_toggle.setText("Diff 표시 ON")
        else:
            self.btn_diff_toggle.setText("Diff 표시 OFF")

        if self.btn_depth.isChecked() and self.last_depth_qimage is not None:
            self.update_recent_depth_view()
        elif self.btn_point.isChecked() and self.last_pointcloud is not None:
            if self.show_diff_overlay and self.compare_pointcloud is not None:
                self.update_recent_pointcloud_diff_view()
            else:
                self.update_recent_pointcloud_view()

    # ===== 카메라 파라미터(게인/노출) 파싱 =====
    def _get_gain_db(self) -> float:
        default = 12.0
        try:
            t = self.edit_gain_db.text().strip()
            if t == "":
                return default
            v = float(t)
            if v < 0:
                raise ValueError
            # 권장 범위(0~16) 안내용 클램프 (원하면 제거 가능)
            if v > 16:
                self.append_log("[WARN] 게인 값이 16dB를 초과했습니다. 16으로 제한합니다.")
                v = 16.0
                self.edit_gain_db.setText("16")
            return v
        except Exception:
            self.append_log("[WARN] 게인 입력이 잘못되어 기본값 12dB를 사용합니다.")
            self.edit_gain_db.setText("12")
            return default

    def _get_exposure_ms(self) -> float:
        default = 6.0
        try:
            t = self.edit_exposure_ms.text().strip()
            if t == "":
                return default
            v = float(t)
            if v <= 0:
                raise ValueError
            return v
        except Exception:
            self.append_log("[WARN] 노출(ms) 입력이 잘못되어 기본값 6.0ms를 사용합니다.")
            self.edit_exposure_ms.setText("6.0")
            return default

    # ===== apply 버튼 슬롯 =====
    def on_apply_camera_params_clicked(self):
        gain_db = self._get_gain_db()
        exp_ms = self._get_exposure_ms()

        self.append_log(f"[INFO] 카메라 파라미터 적용: gain={gain_db:.2f} dB, exposure={exp_ms:.2f} ms (2D/3D 동일)")

        if not USE_REAL_CAMERA:
            self.append_log("[INFO] 샘플 모드(USE_REAL_CAMERA=False) → 실제 카메라 적용은 생략합니다.")
            return

        if not self.camera_connected or self.camera is None:
            self.append_log("[ERROR] 카메라 미연결 상태입니다. 먼저 Connect 하세요.")
            return

        self.apply_camera_parameters(gain_db, exp_ms)

    def apply_camera_parameters(self, gain_db: float, exposure_ms: float):
        """
        실제 카메라 UserSet에 2D/3D 동일 게인/노출 적용
        - 3D ExposureSequence = [exposure_ms]
        - 2D ExposureMode(Timed) + ExposureTime = exposure_ms
        - 2D Gain = gain_db
        - 3D Gain = gain_db (파라미터 이름 후보로 시도)
        """
        try:
            user_set = self.camera.current_user_set()

            # --- 3D Exposure (Sequence) ---
            err = user_set.set_float_array_value(Scanning3DExposureSequence.name, [float(exposure_ms)])
            show_error(err)
            self.append_log(f"[OK] 3D ExposureSequence = [{exposure_ms}] ms")

            # --- 2D ExposureMode / ExposureTime ---
            try:
                err = user_set.set_enum_value(Scanning2DExposureMode.name, Scanning2DExposureMode.Value_Timed)
                show_error(err)
            except Exception as e:
                self.append_log(f"[WARN] 2D ExposureMode(Timed) 설정 실패(모델/SDK 차이): {e}")

            try:
                err = user_set.set_float_value(Scanning2DExposureTime.name, float(exposure_ms))
                show_error(err)
                self.append_log(f"[OK] 2D ExposureTime = {exposure_ms} ms")
            except Exception as e:
                self.append_log(f"[WARN] 2D ExposureTime 설정 실패: {e}")

            # --- 2D Gain ---
            try:
                err = user_set.set_float_value(Scanning2DGain.name, float(gain_db))
                show_error(err)
                self.append_log(f"[OK] 2D Gain = {gain_db} dB")
            except Exception as e:
                self.append_log(f"[WARN] 2D Gain 적용 실패(모델/파라미터 차이): {e}")

            # --- 3D Gain (후보 이름으로 시도) ---
            candidates = []
            try:
                candidates.append(Scanning3DGain.name)  # 존재하면 우선
            except Exception:
                pass
            candidates += ["Scanning3DGain", "Scan3DGain", "Scan3D_Gain"]

            applied = False
            for pname in candidates:
                try:
                    err = user_set.set_float_value(pname, float(gain_db))
                    show_error(err)
                    self.append_log(f"[OK] 3D Gain = {gain_db} dB (param='{pname}')")
                    applied = True
                    break
                except Exception:
                    continue

            if not applied:
                self.append_log("[WARN] 3D Gain 파라미터를 찾지 못했거나 적용 실패했습니다. (Scanning3DGain/Scan3DGain 확인 필요)")

            # --- 저장 ---
            try:
                show_error(user_set.save_all_parameters_to_device(), "\nSave the current parameter settings to the selected user set.")
                self.append_log("[OK] UserSet 저장 완료")
            except Exception as e:
                self.append_log(f"[WARN] UserSet 저장 실패(권한/모델 차이): {e}")

        except Exception as e:
            self.append_log(f"[ERROR] 카메라 파라미터 적용 중 예외: {e}")

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