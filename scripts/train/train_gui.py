from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import json
import re
import sys

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


TRAIN_METRIC_RE = re.compile(r"^\[TRAIN_METRIC\]\s+(.*)$")
KV_RE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")
RUN_DIR_RE = re.compile(r"^\[CONFIG\]\s+run_dir=(.+)$")
RESULT_RE = re.compile(r"^\[RESULT\]\s+([^:]+):\s*(.*)$")
CONFIG_RE = re.compile(r"^\[CONFIG\]\s+([^=]+)=(.*)$")


@dataclass
class UnifiedPipelineConfig:
    config: str = ""

    pipeline: str = "pinn"

    noisy_mat: str = ""
    clean_mat: str = ""
    inference_input: str = ""

    data_dir: str = "data"
    results_dir: str = "results"
    checkpoint: str = ""
    resume_checkpoint: str = ""

    skip_transform: bool = False
    skip_train: bool = False
    skip_inference: bool = False

    input_cols: int = 21
    input_rows: int = 21
    target_cols: int = 41
    target_rows: int = 41
    inference_input_cols: int | None = None
    inference_input_rows: int | None = None
    inference_target_cols: int | None = None
    inference_target_rows: int | None = None
    signal_length: int = 1000
    interp_method: str = "cubic"
    augment_factor: int = 5
    train_ratio: float = 0.8

    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
    patience: int = 50
    min_epochs: int = 30
    dropout: float = 0.1
    physics_weight: float | None = None

    patch_size: int = 5
    stride: int = 1
    model_type: str = "deepsets"
    base_channels: int = 16
    coord_dim: int = 64
    signal_embed_dim: int = 128
    coord_embed_dim: int = 64
    point_dim: int = 128
    tf_embed_dim: int = 128
    stft_n_fft: int = 128
    stft_hop_length: int = 32
    stft_win_length: int = 128
    stft_window: str = "hann"
    stft_pooling: str = "mean"
    fusion_mode: str = "gated"
    debug_numerics: bool = False

    wave_speed: float = 5900.0
    center_frequency: float = 250e3
    damping_ratio: float = 0.05

    inference_batch_size: int = 64
    validation_samples: int = 20

    log_experiment: bool = True
    experiment_dir: str = "experiments"
    experiment_tag: str = ""


class FigurePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.subplots(2, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.clear()

    def clear(self) -> None:
        for ax in self.axes.flatten():
            ax.clear()
            ax.grid(True, alpha=0.3)

        self.axes[0, 0].set_title("Total Loss")
        self.axes[0, 1].set_title("Data/Physics Loss")
        self.axes[1, 0].set_title("PSNR")
        self.axes[1, 1].set_title("Learning Rate")
        self.canvas.draw_idle()

    def update_metrics(self, metrics: list[dict[str, float]]) -> None:
        if not metrics:
            self.clear()
            return

        epochs = [int(m["epoch"]) for m in metrics]

        train_total = [float(m.get("train_total", 0.0)) for m in metrics]
        val_total = [float(m.get("val_total", 0.0)) for m in metrics]
        train_data = [float(m.get("train_data", 0.0)) for m in metrics]
        val_data = [float(m.get("val_data", 0.0)) for m in metrics]
        train_phys = [float(m.get("train_physics", 0.0)) for m in metrics]
        val_phys = [float(m.get("val_physics", 0.0)) for m in metrics]
        train_psnr = [float(m.get("train_psnr", 0.0)) for m in metrics]
        val_psnr = [float(m.get("val_psnr", 0.0)) for m in metrics]
        lr = [float(m.get("lr", 0.0)) for m in metrics]

        for ax in self.axes.flatten():
            ax.clear()
            ax.grid(True, alpha=0.3)

        ax = self.axes[0, 0]
        ax.plot(epochs, train_total, "b-", label="Train Total", linewidth=1.4)
        ax.plot(epochs, val_total, "r-", label="Val Total", linewidth=1.4)
        ax.set_yscale("log")
        ax.set_title("Total Loss")
        ax.legend(fontsize=8)

        ax = self.axes[0, 1]
        ax.plot(epochs, train_data, "b-", label="Train Data", linewidth=1.2)
        ax.plot(epochs, val_data, "r-", label="Val Data", linewidth=1.2)
        ax.plot(epochs, train_phys, "b--", label="Train Physics", linewidth=1.0)
        ax.plot(epochs, val_phys, "r--", label="Val Physics", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title("Data / Physics Loss")
        ax.legend(fontsize=8)

        ax = self.axes[1, 0]
        ax.plot(epochs, train_psnr, "b-", label="Train PSNR", linewidth=1.4)
        ax.plot(epochs, val_psnr, "r-", label="Val PSNR", linewidth=1.4)
        ax.set_title("PSNR (dB)")
        ax.legend(fontsize=8)

        ax = self.axes[1, 1]
        ax.plot(epochs, lr, "m-", label="LR", linewidth=1.4)
        ax.set_yscale("log")
        ax.set_title("Learning Rate")
        ax.legend(fontsize=8)

        self.canvas.draw_idle()


class ImageView(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title

        root = QVBoxLayout(self)
        self.path_label = QLabel("未加载")
        self.path_label.setWordWrap(True)

        self.image_label = QLabel("暂无图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(240)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)

        root.addWidget(QLabel(title))
        root.addWidget(self.path_label)
        root.addWidget(scroll)

    def set_image(self, image_path: Path | None) -> None:
        if image_path is None or not image_path.exists():
            self.path_label.setText("未找到")
            self.image_label.setText("暂无图片")
            self.image_label.setPixmap(QPixmap())
            return

        self.path_label.setText(str(image_path))
        pix = QPixmap(str(image_path))
        if pix.isNull():
            self.image_label.setText("该图片格式无法直接预览，请使用外部工具打开")
            self.image_label.setPixmap(QPixmap())
            return

        scaled = pix.scaledToWidth(980, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)


class TrainingGuiWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Unified Pipeline Training GUI")
        self.resize(1820, 1040)

        self.project_root = Path(__file__).resolve().parents[2]
        self.config_dir = Path(__file__).resolve().parent / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.field_widgets: dict[str, Any] = {}
        self.process: QProcess | None = None
        self.partial_output = ""
        self.metrics: list[dict[str, float]] = []
        self.current_run_config: dict[str, Any] | None = None
        self.queued_config: dict[str, Any] | None = None

        self.current_run_dir: Path | None = None
        self.current_run_log_path: Path | None = None
        self.current_metrics_path: Path | None = None
        self.log_file = None
        self.metrics_file = None
        self.pending_lines_before_log: list[str] = []

        self.result_checkpoint = ""
        self.result_denoised = ""
        self.result_validation = ""
        self.result_run_dir = ""

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = self._build_left_panel()
        right = self._build_right_panel()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([560, 1240])

        self._apply_config_to_ui(UnifiedPipelineConfig())

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        form_group = QGroupBox("参数配置")
        form_layout_outer = QVBoxLayout(form_group)
        form_widget = QWidget()
        form = QFormLayout(form_widget)

        self._add_path_line(form, "config", "配置JSON路径")
        self._add_combo(form, "pipeline", "Pipeline", ["pinn", "deepsets"])

        self._add_path_line(form, "noisy_mat", "noisy_mat")
        self._add_path_line(form, "clean_mat", "clean_mat")
        self._add_path_line(form, "inference_input", "inference_input")

        self._add_line(form, "data_dir", "data_dir")
        self._add_line(form, "results_dir", "results_dir")
        self._add_path_line(form, "checkpoint", "checkpoint")
        self._add_path_line(form, "resume_checkpoint", "resume_checkpoint")

        self._add_check(form, "skip_transform", "skip_transform")
        self._add_check(form, "skip_train", "skip_train")
        self._add_check(form, "skip_inference", "skip_inference")

        self._add_line(form, "input_cols", "input_cols")
        self._add_line(form, "input_rows", "input_rows")
        self._add_line(form, "target_cols", "target_cols")
        self._add_line(form, "target_rows", "target_rows")
        self._add_line(form, "inference_input_cols", "inference_input_cols(可空)")
        self._add_line(form, "inference_input_rows", "inference_input_rows(可空)")
        self._add_line(form, "inference_target_cols", "inference_target_cols(可空)")
        self._add_line(form, "inference_target_rows", "inference_target_rows(可空)")
        self._add_line(form, "signal_length", "signal_length")
        self._add_combo(form, "interp_method", "interp_method", ["linear", "cubic"])
        self._add_line(form, "augment_factor", "augment_factor")
        self._add_line(form, "train_ratio", "train_ratio")

        self._add_line(form, "epochs", "epochs")
        self._add_line(form, "batch_size", "batch_size")
        self._add_line(form, "lr", "lr")
        self._add_line(form, "seed", "seed")
        self._add_line(form, "patience", "patience")
        self._add_line(form, "min_epochs", "min_epochs")
        self._add_line(form, "dropout", "dropout")
        self._add_line(form, "physics_weight", "physics_weight(可空)")

        self._add_line(form, "patch_size", "patch_size")
        self._add_line(form, "stride", "stride")
        self._add_combo(form, "model_type", "model_type", ["deepsets", "tf_fusion"])
        self._add_line(form, "base_channels", "base_channels")
        self._add_line(form, "coord_dim", "coord_dim")
        self._add_line(form, "signal_embed_dim", "signal_embed_dim")
        self._add_line(form, "coord_embed_dim", "coord_embed_dim")
        self._add_line(form, "point_dim", "point_dim")
        self._add_line(form, "tf_embed_dim", "tf_embed_dim")
        self._add_line(form, "stft_n_fft", "stft_n_fft")
        self._add_line(form, "stft_hop_length", "stft_hop_length")
        self._add_line(form, "stft_win_length", "stft_win_length")
        self._add_line(form, "stft_window", "stft_window")
        self._add_combo(form, "stft_pooling", "stft_pooling", ["mean", "max", "meanmax"])
        self._add_combo(form, "fusion_mode", "fusion_mode", ["gated", "concat"])
        self._add_check(form, "debug_numerics", "debug_numerics")

        self._add_line(form, "wave_speed", "wave_speed")
        self._add_line(form, "center_frequency", "center_frequency")
        self._add_line(form, "damping_ratio", "damping_ratio")

        self._add_line(form, "inference_batch_size", "inference_batch_size")
        self._add_line(form, "validation_samples", "validation_samples")

        self._add_check(form, "log_experiment", "log_experiment")
        self._add_line(form, "experiment_dir", "experiment_dir")
        self._add_line(form, "experiment_tag", "experiment_tag")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        form_layout_outer.addWidget(scroll)

        config_name_row = QHBoxLayout()
        config_name_row.addWidget(QLabel("配置文件名:"))
        self.config_name_edit = QLineEdit("pipeline_ui_config")
        self.config_name_edit.setPlaceholderText("例如 my_exp")
        config_name_row.addWidget(self.config_name_edit)
        form_layout_outer.addLayout(config_name_row)

        btn_row1 = QHBoxLayout()
        self.btn_start = QPushButton("开始执行")
        self.btn_start.clicked.connect(self.start_or_schedule_run)
        btn_row1.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止当前任务")
        self.btn_stop.clicked.connect(self.stop_current_run)
        btn_row1.addWidget(self.btn_stop)

        self.btn_resume_pick = QPushButton("选择续训Checkpoint")
        self.btn_resume_pick.clicked.connect(self.pick_resume_checkpoint)
        btn_row1.addWidget(self.btn_resume_pick)
        form_layout_outer.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_save_cfg = QPushButton("保存超参数配置")
        self.btn_save_cfg.clicked.connect(self.save_config_file)
        btn_row2.addWidget(self.btn_save_cfg)

        self.btn_load_cfg = QPushButton("加载超参数配置")
        self.btn_load_cfg.clicked.connect(self.load_config_file)
        btn_row2.addWidget(self.btn_load_cfg)
        form_layout_outer.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        self.btn_load_log = QPushButton("加载日志回放")
        self.btn_load_log.clicked.connect(self.load_log_and_replay)
        btn_row3.addWidget(self.btn_load_log)

        self.btn_open_run_dir = QPushButton("打开最近run目录")
        self.btn_open_run_dir.clicked.connect(self.pick_run_dir)
        btn_row3.addWidget(self.btn_open_run_dir)
        form_layout_outer.addLayout(btn_row3)

        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        form_layout_outer.addWidget(self.status_label)

        layout.addWidget(form_group)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tabs = QTabWidget()

        self.fig_panel = FigurePanel()
        self.tabs.addTab(self.fig_panel, "实时训练曲线")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        self.tabs.addTab(log_tab, "控制台日志")

        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        self.training_image = ImageView("训练曲线图")
        self.validation_image = ImageView("声学验证图")
        self.inference_image = ImageView("推理结果对比图")

        image_layout.addWidget(self.training_image)
        image_layout.addWidget(self.validation_image)
        image_layout.addWidget(self.inference_image)
        self.tabs.addTab(image_tab, "训练结果图")

        summary_tab = QWidget()
        summary_layout = QFormLayout(summary_tab)
        self.run_dir_label = QLabel("-")
        self.run_dir_label.setWordWrap(True)
        self.checkpoint_label = QLabel("-")
        self.checkpoint_label.setWordWrap(True)
        self.denoised_label = QLabel("-")
        self.denoised_label.setWordWrap(True)
        self.validation_label = QLabel("-")
        self.validation_label.setWordWrap(True)
        self.log_path_label = QLabel("-")
        self.log_path_label.setWordWrap(True)
        self.metrics_path_label = QLabel("-")
        self.metrics_path_label.setWordWrap(True)

        summary_layout.addRow("run_dir", self.run_dir_label)
        summary_layout.addRow("checkpoint", self.checkpoint_label)
        summary_layout.addRow("denoised_mat", self.denoised_label)
        summary_layout.addRow("validation_figure", self.validation_label)
        summary_layout.addRow("run.log", self.log_path_label)
        summary_layout.addRow("metrics.jsonl", self.metrics_path_label)
        self.tabs.addTab(summary_tab, "运行摘要")

        layout.addWidget(self.tabs)
        return panel

    def _add_line(self, form: QFormLayout, key: str, label: str) -> None:
        edit = QLineEdit()
        form.addRow(label, edit)
        self.field_widgets[key] = edit

    def _add_check(self, form: QFormLayout, key: str, label: str) -> None:
        check = QCheckBox()
        form.addRow(label, check)
        self.field_widgets[key] = check

    def _add_combo(self, form: QFormLayout, key: str, label: str, options: list[str]) -> None:
        combo = QComboBox()
        combo.addItems(options)
        form.addRow(label, combo)
        self.field_widgets[key] = combo

    def _add_path_line(self, form: QFormLayout, key: str, label: str) -> None:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)

        edit = QLineEdit()
        btn = QPushButton("...")
        btn.setMaximumWidth(36)
        btn.clicked.connect(lambda: self._pick_path_for_key(key))

        row.addWidget(edit)
        row.addWidget(btn)
        form.addRow(label, row_widget)
        self.field_widgets[key] = edit

    def _pick_path_for_key(self, key: str) -> None:
        default_dir = str(self.project_root)
        if key in {"data_dir", "results_dir", "experiment_dir"}:
            path = QFileDialog.getExistingDirectory(self, "选择目录", default_dir)
            if path:
                widget = self.field_widgets[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(path)
            return

        path, _ = QFileDialog.getOpenFileName(self, "选择文件", default_dir)
        if path:
            widget = self.field_widgets[key]
            if isinstance(widget, QLineEdit):
                widget.setText(path)

    def _parse_optional_int(self, raw: str) -> int | None:
        value = raw.strip()
        if value == "":
            return None
        return int(value)

    def _parse_optional_float(self, raw: str) -> float | None:
        value = raw.strip()
        if value == "":
            return None
        return float(value)

    def _collect_config_from_ui(self) -> UnifiedPipelineConfig:
        cfg = UnifiedPipelineConfig()

        for key, widget in self.field_widgets.items():
            if isinstance(widget, QCheckBox):
                setattr(cfg, key, bool(widget.isChecked()))
            elif isinstance(widget, QComboBox):
                setattr(cfg, key, widget.currentText())
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                setattr(cfg, key, text)

        int_keys = {
            "input_cols",
            "input_rows",
            "target_cols",
            "target_rows",
            "signal_length",
            "augment_factor",
            "epochs",
            "batch_size",
            "seed",
            "patience",
            "min_epochs",
            "patch_size",
            "stride",
            "base_channels",
            "coord_dim",
            "signal_embed_dim",
            "coord_embed_dim",
            "point_dim",
            "tf_embed_dim",
            "stft_n_fft",
            "stft_hop_length",
            "stft_win_length",
            "inference_batch_size",
            "validation_samples",
        }

        float_keys = {
            "train_ratio",
            "lr",
            "dropout",
            "wave_speed",
            "center_frequency",
            "damping_ratio",
        }

        optional_int_keys = {
            "inference_input_cols",
            "inference_input_rows",
            "inference_target_cols",
            "inference_target_rows",
        }

        for key in int_keys:
            raw = getattr(cfg, key)
            if raw == "":
                raise ValueError(f"{key} 不能为空")
            setattr(cfg, key, int(raw))

        for key in float_keys:
            raw = getattr(cfg, key)
            if raw == "":
                raise ValueError(f"{key} 不能为空")
            setattr(cfg, key, float(raw))

        for key in optional_int_keys:
            raw = getattr(cfg, key)
            setattr(cfg, key, self._parse_optional_int(raw))

        cfg.physics_weight = self._parse_optional_float(str(getattr(cfg, "physics_weight")))

        cfg.config = str(getattr(cfg, "config")).strip()
        cfg.noisy_mat = str(getattr(cfg, "noisy_mat")).strip()
        cfg.clean_mat = str(getattr(cfg, "clean_mat")).strip()
        cfg.inference_input = str(getattr(cfg, "inference_input")).strip()
        cfg.data_dir = str(getattr(cfg, "data_dir")).strip() or "data"
        cfg.results_dir = str(getattr(cfg, "results_dir")).strip() or "results"
        cfg.checkpoint = str(getattr(cfg, "checkpoint")).strip()
        cfg.resume_checkpoint = str(getattr(cfg, "resume_checkpoint")).strip()
        cfg.stft_window = str(getattr(cfg, "stft_window")).strip() or "hann"
        cfg.experiment_dir = str(getattr(cfg, "experiment_dir")).strip() or "experiments"
        cfg.experiment_tag = str(getattr(cfg, "experiment_tag")).strip()

        return cfg

    def _apply_config_to_ui(self, cfg: UnifiedPipelineConfig) -> None:
        data = asdict(cfg)
        for key, widget in self.field_widgets.items():
            value = data.get(key)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QLineEdit):
                widget.setText("" if value is None else str(value))

    def pick_resume_checkpoint(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择续训 checkpoint",
            str((self.project_root / "results").resolve()),
            "Checkpoint (*.pth);;All Files (*)",
        )
        if not file_path:
            return

        widget = self.field_widgets.get("resume_checkpoint")
        if not isinstance(widget, QLineEdit):
            return
        resume_edit = cast(QLineEdit, widget)
        resume_edit.setText(file_path)

    def _build_command(self, cfg: UnifiedPipelineConfig) -> list[str]:
        args = ["run", "python", "scripts/run_unified_pipeline.py"]
        data = asdict(cfg)

        key_order = [
            "config",
            "pipeline",
            "noisy_mat",
            "clean_mat",
            "inference_input",
            "data_dir",
            "results_dir",
            "checkpoint",
            "resume_checkpoint",
            "skip_transform",
            "skip_train",
            "skip_inference",
            "input_cols",
            "input_rows",
            "target_cols",
            "target_rows",
            "inference_input_cols",
            "inference_input_rows",
            "inference_target_cols",
            "inference_target_rows",
            "signal_length",
            "interp_method",
            "augment_factor",
            "train_ratio",
            "epochs",
            "batch_size",
            "lr",
            "seed",
            "patience",
            "min_epochs",
            "dropout",
            "physics_weight",
            "patch_size",
            "stride",
            "model_type",
            "base_channels",
            "coord_dim",
            "signal_embed_dim",
            "coord_embed_dim",
            "point_dim",
            "tf_embed_dim",
            "stft_n_fft",
            "stft_hop_length",
            "stft_win_length",
            "stft_window",
            "stft_pooling",
            "fusion_mode",
            "debug_numerics",
            "wave_speed",
            "center_frequency",
            "damping_ratio",
            "inference_batch_size",
            "validation_samples",
            "log_experiment",
            "experiment_dir",
            "experiment_tag",
        ]

        bool_keys = {
            "skip_transform",
            "skip_train",
            "skip_inference",
            "debug_numerics",
            "log_experiment",
        }

        for key in key_order:
            value = data.get(key)
            if key in bool_keys:
                if bool(value):
                    args.append(f"--{key}")
                continue

            if value is None:
                continue
            if isinstance(value, str) and value.strip() == "":
                continue

            args.append(f"--{key}")
            args.append(str(value))

        return args

    def start_or_schedule_run(self) -> None:
        try:
            cfg = self._collect_config_from_ui()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"参数解析失败:\n{exc}")
            return

        config_dict = asdict(cfg)

        if self.process is not None and self.process.state() != QProcess.NotRunning:
            msg = QMessageBox(self)
            msg.setWindowTitle("已有任务在运行")
            msg.setText("当前已有任务运行中，请选择操作")
            terminate_btn = msg.addButton("终止当前并执行新任务", QMessageBox.AcceptRole)
            queue_btn = msg.addButton("排队等待", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("取消", QMessageBox.RejectRole)
            msg.exec()
            clicked = msg.clickedButton()

            if clicked == cancel_btn:
                return
            if clicked == queue_btn:
                self.queued_config = config_dict
                self.status_label.setText("任务已排队，当前任务结束后自动开始")
                return
            if clicked == terminate_btn:
                self.queued_config = config_dict
                self.stop_current_run()
                return

        self._start_run(config_dict)

    def _start_run(self, config_dict: dict[str, Any]) -> None:
        self._reset_runtime_state_for_new_run()

        self.current_run_config = config_dict
        cmd = self._build_command(UnifiedPipelineConfig(**config_dict))

        proc = QProcess(self)
        self.process = proc
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(str(self.project_root))
        proc.readyReadStandardOutput.connect(self._on_process_output)
        proc.finished.connect(self._on_process_finished)

        self.status_label.setText("任务启动中...")
        self.log_text.appendPlainText("$ uv " + " ".join(cmd))

        proc.start("uv", cmd)
        if not proc.waitForStarted(5000):
            QMessageBox.critical(self, "错误", "无法启动任务，请检查 uv 环境")
            self.status_label.setText("启动失败")
            self.process = None
            return

        self.status_label.setText("任务运行中")

    def stop_current_run(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.status_label.setText("当前没有运行中的任务")
            return

        self.status_label.setText("正在尝试停止任务...")
        self.process.terminate()

        def _force_kill() -> None:
            if self.process is not None and self.process.state() != QProcess.NotRunning:
                self.process.kill()

        QTimer.singleShot(3000, _force_kill)

    def _on_process_output(self) -> None:
        if self.process is None:
            return

        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not chunk:
            return

        data = self.partial_output + chunk
        lines = data.split("\n")
        self.partial_output = lines[-1]

        for line in lines[:-1]:
            clean_line = line.rstrip("\r")
            self._handle_output_line(clean_line)

    def _handle_output_line(self, line: str) -> None:
        self.log_text.appendPlainText(line)

        if self.current_run_log_path is None:
            self.pending_lines_before_log.append(line)
        else:
            self._append_run_log_line(line)

        self._try_resolve_run_dir(line)
        self._try_parse_metric_line(line)
        self._try_parse_result_line(line)

    def _try_resolve_run_dir(self, line: str) -> None:
        if self.current_run_dir is not None:
            return

        m = RUN_DIR_RE.match(line)
        if m is None:
            return

        run_dir = Path(m.group(1).strip())
        self.current_run_dir = run_dir
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_log_path = logs_dir / "run.log"
        self.current_metrics_path = logs_dir / "metrics.jsonl"

        self.log_file = self.current_run_log_path.open("a", encoding="utf-8")
        self.metrics_file = self.current_metrics_path.open("a", encoding="utf-8")

        for cached in self.pending_lines_before_log:
            self._append_run_log_line(cached)
        self.pending_lines_before_log.clear()

        self.run_dir_label.setText(str(run_dir))
        self.log_path_label.setText(str(self.current_run_log_path))
        self.metrics_path_label.setText(str(self.current_metrics_path))

        if self.current_run_config is not None:
            hparams_path = logs_dir / "hparams_snapshot.json"
            hparams_path.write_text(
                json.dumps(self.current_run_config, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            cmd_path = logs_dir / "launch_command.txt"
            cmd = self._build_command(UnifiedPipelineConfig(**self.current_run_config))
            cmd_path.write_text("uv " + " ".join(cmd) + "\n", encoding="utf-8")

    def _try_parse_metric_line(self, line: str) -> None:
        m = TRAIN_METRIC_RE.match(line)
        if m is None:
            return

        payload = m.group(1)
        pairs = dict(KV_RE.findall(payload))
        if "epoch" not in pairs:
            return

        metric: dict[str, float] = {}
        for key, value in pairs.items():
            if key == "pipeline":
                continue
            try:
                metric[key] = float(value)
            except ValueError:
                continue

        if "epoch" not in metric:
            return

        self.metrics.append(metric)
        self.fig_panel.update_metrics(self.metrics)

        if self.metrics_file is not None:
            self.metrics_file.write(json.dumps(metric, ensure_ascii=False) + "\n")
            self.metrics_file.flush()

        epoch = int(metric.get("epoch", 0))
        val_psnr = metric.get("val_psnr", float("nan"))
        self.status_label.setText(f"训练中：epoch={epoch}, val_psnr={val_psnr:.3f} dB")

    def _try_parse_result_line(self, line: str) -> None:
        m = RESULT_RE.match(line)
        if m is None:
            return

        key = m.group(1).strip().lower()
        value = m.group(2).strip()

        if key == "checkpoint":
            self.result_checkpoint = value
            self.checkpoint_label.setText(value)
        elif key == "denoised mat":
            self.result_denoised = value
            self.denoised_label.setText(value)
        elif key == "acoustic validation figure":
            self.result_validation = value
            self.validation_label.setText(value)
        elif key == "run directory":
            self.result_run_dir = value
            self.run_dir_label.setText(value)

    def _append_run_log_line(self, line: str) -> None:
        if self.log_file is None:
            return
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self.partial_output:
            self._handle_output_line(self.partial_output.rstrip("\r"))
            self.partial_output = ""

        self._close_log_handles()

        if exit_code == 0:
            self.status_label.setText("任务结束：成功")
        else:
            self.status_label.setText(f"任务结束：失败 (code={exit_code})")

        if self.current_run_dir is not None:
            self._load_result_images(self.current_run_dir)

        self.process = None

        if self.queued_config is not None:
            queued = self.queued_config
            self.queued_config = None
            self.status_label.setText("开始执行排队任务...")
            self._start_run(queued)

    def _reset_runtime_state_for_new_run(self) -> None:
        self.partial_output = ""
        self.metrics = []
        self.fig_panel.clear()
        self.log_text.clear()

        self.current_run_dir = None
        self.current_run_log_path = None
        self.current_metrics_path = None
        self.pending_lines_before_log = []
        self.result_checkpoint = ""
        self.result_denoised = ""
        self.result_validation = ""
        self.result_run_dir = ""

        self.run_dir_label.setText("-")
        self.checkpoint_label.setText("-")
        self.denoised_label.setText("-")
        self.validation_label.setText("-")
        self.log_path_label.setText("-")
        self.metrics_path_label.setText("-")

        self.training_image.set_image(None)
        self.validation_image.set_image(None)
        self.inference_image.set_image(None)

        self._close_log_handles()

    def _close_log_handles(self) -> None:
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.metrics_file is not None:
            self.metrics_file.close()
            self.metrics_file = None

    def _load_result_images(self, run_dir: Path) -> None:
        img_dir = run_dir / "images"
        if not img_dir.exists():
            self.training_image.set_image(None)
            self.validation_image.set_image(None)
            self.inference_image.set_image(None)
            return

        train_curve = self._find_first(
            img_dir,
            [
                "*training_curves*.png",
                "*training_curves*.jpg",
                "*training_curves*.jpeg",
            ],
        )
        validation = self._find_first(
            img_dir,
            ["*acoustic_validation*.png", "*acoustic_validation*.jpg", "*acoustic_validation*.jpeg"],
        )
        infer_cmp = self._find_first(img_dir, ["fig_inferenced.png", "*inferenced*.png"])

        self.training_image.set_image(train_curve)
        self.validation_image.set_image(validation)
        self.inference_image.set_image(infer_cmp)

    def _find_first(self, root: Path, patterns: list[str]) -> Path | None:
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
        return None

    def save_config_file(self) -> None:
        try:
            cfg = self._collect_config_from_ui()
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"参数解析失败:\n{exc}")
            return

        raw_name = self.config_name_edit.text().strip()
        if not raw_name:
            QMessageBox.warning(self, "提示", "请填写配置文件名")
            return

        filename = raw_name if raw_name.endswith(".json") else f"{raw_name}.json"
        output = self.config_dir / filename

        if output.exists():
            answer = QMessageBox.question(
                self,
                "覆盖确认",
                f"配置文件已存在，是否覆盖？\n{output}",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        output.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_label.setText(f"配置已保存: {output}")

    def load_config_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择配置文件",
            str(self.config_dir),
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            cfg = UnifiedPipelineConfig()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            self._apply_config_to_ui(cfg)
            self.status_label.setText(f"配置已加载: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"加载配置失败:\n{exc}")

    def load_log_and_replay(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择日志或指标文件",
            str((self.project_root / "results").resolve()),
            "Log Files (*.log *.jsonl *.json)",
        )
        if not path:
            return

        selected = Path(path)
        try:
            self.metrics = []
            self.fig_panel.clear()

            run_dir = self._infer_run_dir_from_selected_file(selected)
            if run_dir is not None:
                self.current_run_dir = run_dir
                self.run_dir_label.setText(str(run_dir))
                self._load_result_images(run_dir)

            if selected.suffix.lower() == ".jsonl":
                for line in selected.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "epoch" in row:
                        self.metrics.append({k: float(v) for k, v in row.items()})
            elif selected.suffix.lower() == ".json":
                data = json.loads(selected.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    cfg = UnifiedPipelineConfig()
                    for k, v in data.items():
                        if hasattr(cfg, k):
                            setattr(cfg, k, v)
                    self._apply_config_to_ui(cfg)
            else:
                config_pairs: dict[str, str] = {}
                for raw_line in selected.read_text(encoding="utf-8", errors="replace").splitlines():
                    self._try_parse_metric_line(raw_line)
                    self._try_parse_result_line(raw_line)

                    cm = CONFIG_RE.match(raw_line)
                    if cm is not None:
                        config_pairs[cm.group(1).strip()] = cm.group(2).strip()

                if not self.metrics:
                    self.metrics = self._collect_metrics_from_log(selected)

                self._try_apply_config_from_pairs(config_pairs)

            if self.metrics:
                self.fig_panel.update_metrics(self.metrics)

            if run_dir is not None:
                self._try_load_hparams_snapshot(run_dir)

            self.status_label.setText(f"日志回放完成: {selected}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"日志回放失败:\n{exc}")

    def _collect_metrics_from_log(self, log_path: Path) -> list[dict[str, float]]:
        collected: list[dict[str, float]] = []
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = TRAIN_METRIC_RE.match(line)
            if m is None:
                continue
            payload = m.group(1)
            pairs = dict(KV_RE.findall(payload))
            if "epoch" not in pairs:
                continue
            metric: dict[str, float] = {}
            for key, value in pairs.items():
                if key == "pipeline":
                    continue
                try:
                    metric[key] = float(value)
                except ValueError:
                    continue
            if "epoch" in metric:
                collected.append(metric)
        return collected

    def _infer_run_dir_from_selected_file(self, selected: Path) -> Path | None:
        if selected.name == "run.log" and selected.parent.name == "logs":
            return selected.parent.parent
        if selected.name == "metrics.jsonl" and selected.parent.name == "logs":
            return selected.parent.parent
        if selected.name == "hparams_snapshot.json" and selected.parent.name == "logs":
            return selected.parent.parent
        if selected.is_dir():
            return selected
        return None

    def _try_load_hparams_snapshot(self, run_dir: Path) -> None:
        hparams = run_dir / "logs" / "hparams_snapshot.json"
        if not hparams.exists():
            return
        data = json.loads(hparams.read_text(encoding="utf-8"))
        cfg = UnifiedPipelineConfig()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        self._apply_config_to_ui(cfg)

    def _try_apply_config_from_pairs(self, config_pairs: dict[str, str]) -> None:
        if not config_pairs:
            return

        cfg = UnifiedPipelineConfig()

        for key, value in config_pairs.items():
            k = key.strip()
            if not hasattr(cfg, k):
                continue

            current = getattr(cfg, k)
            try:
                if isinstance(current, bool):
                    setattr(cfg, k, value.lower() in {"1", "true", "yes"})
                elif isinstance(current, int):
                    setattr(cfg, k, int(value))
                elif isinstance(current, float):
                    setattr(cfg, k, float(value))
                else:
                    setattr(cfg, k, value)
            except Exception:
                continue

        self._apply_config_to_ui(cfg)

    def pick_run_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择 run 目录",
            str((self.project_root / "results").resolve()),
        )
        if not path:
            return

        run_dir = Path(path)
        self.current_run_dir = run_dir
        self.run_dir_label.setText(str(run_dir))
        self._load_result_images(run_dir)

        metrics_path = run_dir / "logs" / "metrics.jsonl"
        if metrics_path.exists():
            self.metrics = []
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.metrics.append({k: float(v) for k, v in row.items()})
            self.fig_panel.update_metrics(self.metrics)
            self.metrics_path_label.setText(str(metrics_path))

        run_log = run_dir / "logs" / "run.log"
        if run_log.exists():
            self.log_path_label.setText(str(run_log))
            self.log_text.setPlainText(run_log.read_text(encoding="utf-8", errors="replace"))

        self._try_load_hparams_snapshot(run_dir)
        self.status_label.setText(f"已加载 run 目录: {run_dir}")


def main() -> None:
    app = QApplication(sys.argv)
    win = TrainingGuiWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
