import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
import threading

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QSlider, QTabWidget, QGroupBox, QGridLayout,
                             QSpinBox, QTextEdit, QSplitter, QFrame,
                             QScrollArea, QSizePolicy, QMessageBox, QCheckBox,
                             QProgressBar)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, Slot, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

# Import YOLO (assuming you have ultralytics installed)
try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not found. Install with 'pip install ultralytics'")
    YOLO = None


class SaveQueue:
    """Queue system for saving images and videos"""

    def __init__(self):
        self.save_queue = Queue()
        self.is_processing = False
        self.worker_thread = None

    def add_to_queue(self, item_type, data, output_path, filename):
        """Add item to save queue"""
        self.save_queue.put({
            'type': item_type,  # 'image' or 'video_frame'
            'data': data,
            'output_path': output_path,
            'filename': filename
        })

        # Start worker thread if not running
        if not self.is_processing:
            self.start_worker()

    def start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.is_processing = True
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()

    def _process_queue(self):
        """Process items in the save queue"""
        while not self.save_queue.empty():
            try:
                item = self.save_queue.get()
                self._save_item(item)
                self.save_queue.task_done()
            except Exception as e:
                print(f"Error saving item: {e}")

        self.is_processing = False

    def _save_item(self, item):
        """Save individual item"""
        try:
            full_path = os.path.join(item['output_path'], item['filename'])

            if item['type'] == 'image':
                cv2.imwrite(full_path, item['data'])
                print(f"Saved image: {item['filename']}")
            elif item['type'] == 'video_frame':
                cv2.imwrite(full_path, item['data'])
                print(f"Saved video frame: {item['filename']}")

        except Exception as e:
            print(f"Error saving {item['filename']}: {e}")


class BatchImageProcessor(QThread):
    """Thread for processing all images in batch for detection and saving"""
    progress_update = Signal(int, int)  # current, total
    processing_complete = Signal(int)  # total processed
    error_occurred = Signal(str)

    def __init__(self, detector, image_files, output_path, save_queue):
        super().__init__()
        self.detector = detector
        self.image_files = image_files
        self.output_path = output_path
        self.save_queue = save_queue
        self.is_running = False

    def run(self):
        """Process all images in batch"""
        try:
            self.is_running = True
            processed_count = 0
            total_images = len(self.image_files)

            for i, image_path in enumerate(self.image_files):
                if not self.is_running:
                    break

                # Load image
                frame = cv2.imread(image_path)
                if frame is None:
                    continue

                # Detect objects
                detected_frame = self.detector.detect(frame) if self.detector else frame.copy()

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                base_name = Path(image_path).stem
                filename = f"{base_name}_detected_{timestamp}_{i:04d}.jpg"

                # Add to save queue
                self.save_queue.add_to_queue('image', detected_frame, self.output_path, filename)
                processed_count += 1

                # Update progress
                self.progress_update.emit(i + 1, total_images)

                # Small delay to prevent overwhelming the system
                time.sleep(0.01)

            self.processing_complete.emit(processed_count)

        except Exception as e:
            self.error_occurred.emit(f"Batch processing error: {str(e)}")

    def stop(self):
        """Stop batch processing"""
        self.is_running = False
        self.wait()


class YOLODetector:
    """YOLO model wrapper for object detection"""

    def __init__(self, model_path):
        if YOLO is None:
            raise ImportError("ultralytics package not found")
        self.model = YOLO(model_path)
        self.confidence = 0.5

    def detect(self, frame):
        """Run detection on frame and return annotated result"""
        results = self.model(frame, conf=self.confidence, verbose=False)
        if results and results[0]:
            return results[0].plot()
        return frame

    def set_confidence(self, confidence):
        """Set confidence threshold"""
        self.confidence = confidence


class DetectionThread(QThread):
    """Thread for running YOLO detection without blocking GUI"""
    frame_ready = Signal(np.ndarray, np.ndarray)  # original, detected
    fps_update = Signal(float)  # current fps
    error_occurred = Signal(str)
    fps_changed = Signal(int)  # signal for fps change
    saving_completed = Signal(str)  # signal when video saving is complete

    def __init__(self, detector, source_path, playback_fps=30, save_video=False, output_path=None, save_queue=None,
                 save_all_detected=False):
        super().__init__()
        self.detector = detector
        self.source_path = source_path
        self.playback_fps = playback_fps  # FPS for playback speed
        self.original_fps = 30  # Will be set from video properties
        self.save_video = save_video
        self.output_path = output_path
        self.save_queue = save_queue
        self.save_all_detected = save_all_detected  # New flag for saving all detected frames
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.video_writer = None
        self.source_type = self._detect_source_type()

        # FPS tracking for display
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

        # FPS change handling
        self.fps_changed.connect(self.update_playback_fps)

        # Video control variables
        self.current_frame_index = 0
        self.total_frames = 0
        self.seek_requested = False
        self.seek_target_frame = 0

        # Video saving control - IMPORTANT: These prevent multiple saves
        self.is_saving_complete = False
        self.save_start_frame = 0
        self.processed_frames_for_save = set()  # Track which frames were saved
        self.video_save_path = None

        # Image saving variables
        self.saved_image_count = 0

    def _detect_source_type(self):
        """Detect if source is image or video"""
        if not os.path.exists(self.source_path):
            return None
        ext = Path(self.source_path).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            return 'image'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return 'video'
        return None

    def run(self):
        try:
            self.is_running = True

            if self.source_type == 'image':
                self.process_image()
            elif self.source_type == 'video':
                self.process_video()
            else:
                self.error_occurred.emit("Unsupported file format")

        except Exception as e:
            self.error_occurred.emit(f"Detection error: {str(e)}")
        finally:
            self.cleanup()

    def process_image(self):
        """Process single image and save if requested"""
        frame = cv2.imread(self.source_path)
        if frame is None:
            self.error_occurred.emit("Could not load image")
            return

        detected_frame = self.detector.detect(frame) if self.detector else frame.copy()
        self.frame_ready.emit(frame, detected_frame)

        # Note: Individual image saving is now handled by batch processing
        # This prevents duplicate saves when using the "Save All Images" feature

    def seek_to_frame(self, frame_number):
        """Seek to specific frame in video"""
        if self.cap and self.source_type == 'video':
            self.seek_requested = True
            self.seek_target_frame = frame_number

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_index = frame_number

            # Read and emit the frame at the new position
            ret, frame = self.cap.read()
            if ret:
                detected_frame = self.detector.detect(frame) if self.detector else frame.copy()
                self.frame_ready.emit(frame, detected_frame)

            self.seek_requested = False

    def process_video(self):
        """Process video with separate playback and save FPS control"""
        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            self.error_occurred.emit("Could not open video")
            return

        # Get original video properties
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.original_fps <= 0:
            self.original_fps = 30  # Default fallback

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if saving is enabled (ONE TIME ONLY)
        if self.save_video and self.output_path and not self.is_saving_complete:
            self.setup_video_writer()

        # Calculate intervals
        playback_interval = 1.0 / self.playback_fps
        save_interval = 1.0 / self.original_fps

        # Timing variables
        self.fps_start_time = time.time()
        self.fps_counter = 0
        last_save_time = time.time()

        # Main processing loop
        while self.cap.isOpened() and self.is_running:
            start_time = time.time()

            # Handle pause
            while self.is_paused and self.is_running:
                time.sleep(0.1)

            if not self.is_running:
                break

            # Handle seeking
            if self.seek_requested:
                time.sleep(0.01)  # Small delay during seek
                continue

            ret, frame = self.cap.read()
            if not ret:
                # End of video - complete saving if it was in progress
                if self.video_writer and not self.is_saving_complete:
                    self.complete_video_save()
                break

            self.current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            # Process frame for detection
            detected_frame = self.detector.detect(frame) if self.detector else frame.copy()
            self.frame_ready.emit(frame, detected_frame)

            # Save frame logic - ONLY SAVE EACH FRAME ONCE
            current_time = time.time()
            if (self.video_writer and
                    not self.is_saving_complete and
                    self.current_frame_index not in self.processed_frames_for_save and
                    (current_time - last_save_time) >= save_interval):
                self.video_writer.write(detected_frame)
                self.processed_frames_for_save.add(self.current_frame_index)
                last_save_time = current_time

            # Update FPS counter for display
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
                self.fps_update.emit(self.current_fps)
                self.fps_counter = 0
                self.fps_start_time = time.time()

            # Playback FPS control (affects display speed only)
            elapsed = time.time() - start_time
            sleep_time = playback_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cap.release()

    def setup_video_writer(self):
        """Setup video writer for saving detected video at original FPS - ONE TIME ONLY"""
        try:
            if self.is_saving_complete:
                return

            # Get original video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(self.source_path).stem
            output_filename = f"{base_name}_detected_{timestamp}.mp4"
            self.video_save_path = os.path.join(self.output_path, output_filename)

            # Initialize video writer with ORIGINAL FPS (not playback FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_save_path, fourcc, self.original_fps, (width, height))

            print(f"Video writer initialized: Original FPS = {self.original_fps}, Playback FPS = {self.playback_fps}")
            print(f"Saving to: {self.video_save_path}")

        except Exception as e:
            self.error_occurred.emit(f"Error setting up video writer: {str(e)}")

    def complete_video_save(self):
        """Complete video saving process"""
        if self.video_writer and not self.is_saving_complete:
            self.video_writer.release()
            self.video_writer = None
            self.is_saving_complete = True
            if self.video_save_path:
                self.saving_completed.emit(f"Video saved: {Path(self.video_save_path).name}")
            print("Video saving completed")

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        # Complete any ongoing save before stopping
        if self.video_writer and not self.is_saving_complete:
            self.complete_video_save()

        self.is_running = False
        self.wait()

    @Slot(int)
    def update_playback_fps(self, new_playback_fps):
        """Update playback FPS dynamically (doesn't affect save FPS)"""
        self.playback_fps = new_playback_fps
        print(f"Playback FPS updated to: {new_playback_fps} (Save FPS remains: {self.original_fps})")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.video_writer and not self.is_saving_complete:
            self.complete_video_save()


class ZoomableLabel(QLabel):
    """Label widget that supports zoom and pan functionality"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f5f5f5;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)

        # Zoom and pan variables
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.last_pan_point = None
        self.original_pixmap = None

        # Enable mouse tracking
        self.setMouseTracking(True)

    def set_image(self, cv_image):
        """Set image from OpenCV format"""
        if cv_image is None:
            return

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.original_pixmap = QPixmap.fromImage(qt_image)
        self.update_display()

    def update_display(self):
        """Update the displayed image with current zoom and pan"""
        if self.original_pixmap is None:
            self.setText("No image loaded")
            return

        # Calculate scaled size
        scaled_size = self.original_pixmap.size() * self.zoom_factor

        # Create scaled pixmap
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Apply pan offset (crop if zoomed in)
        if self.zoom_factor > 1.0:
            label_size = self.size()
            x_offset = max(0, min(self.pan_offset[0], scaled_size.width() - label_size.width()))
            y_offset = max(0, min(self.pan_offset[1], scaled_size.height() - label_size.height()))

            crop_rect = scaled_pixmap.rect()
            crop_rect.setX(x_offset)
            crop_rect.setY(y_offset)
            crop_rect.setWidth(min(label_size.width(), scaled_size.width() - x_offset))
            crop_rect.setHeight(min(label_size.height(), scaled_size.height() - y_offset))

            scaled_pixmap = scaled_pixmap.copy(crop_rect)

        self.setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if self.original_pixmap is None:
            return

        # Get mouse position
        mouse_pos = event.position().toPoint()

        # Calculate zoom
        zoom_in = event.angleDelta().y() > 0
        zoom_delta = 1.2 if zoom_in else 1 / 1.2

        old_zoom = self.zoom_factor
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor * zoom_delta))

        if self.zoom_factor != old_zoom:
            # Adjust pan to keep mouse position fixed
            zoom_ratio = self.zoom_factor / old_zoom
            self.pan_offset[0] = int(self.pan_offset[0] * zoom_ratio + mouse_pos.x() * (zoom_ratio - 1))
            self.pan_offset[1] = int(self.pan_offset[1] * zoom_ratio + mouse_pos.y() * (zoom_ratio - 1))

            self.update_display()

    def mousePressEvent(self, event):
        """Start panning"""
        if event.button() == Qt.LeftButton:
            self.last_pan_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        """Handle panning"""
        if self.last_pan_point is not None and self.zoom_factor > 1.0:
            delta = event.position().toPoint() - self.last_pan_point
            self.pan_offset[0] -= delta.x()
            self.pan_offset[1] -= delta.y()
            self.last_pan_point = event.position().toPoint()
            self.update_display()

    def mouseReleaseEvent(self, event):
        """End panning"""
        if event.button() == Qt.LeftButton:
            self.last_pan_point = None

    def reset_view(self):
        """Reset zoom and pan"""
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        self.update_display()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.detection_thread = None
        self.batch_processor = None
        self.current_file_list = []
        self.current_file_index = 0
        self.current_original_frame = None
        self.current_detected_frame = None

        # Initialize save queue
        self.save_queue = SaveQueue()

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        self.setWindowTitle("INDIAN ARMY OBJECT DETECTION MODEL")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget with splitter for resizable panels
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Main horizontal splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)

        # Left control panel
        left_panel = self.create_control_panel()
        self.main_splitter.addWidget(left_panel)

        # Right display panel
        right_panel = self.create_display_panel()
        self.main_splitter.addWidget(right_panel)

        # Set initial splitter sizes (1:2 ratio)
        self.main_splitter.setSizes([400, 800])
        self.main_splitter.setCollapsible(0, False)  # Don't allow control panel to collapse
        self.main_splitter.setCollapsible(1, False)  # Don't allow display panel to collapse

    def create_control_panel(self):
        """Create left control panel"""
        panel = QWidget()
        panel.setMinimumWidth(350)
        panel.setMaximumWidth(500)
        layout = QVBoxLayout(panel)

        # Logo and Title
        header_layout = QVBoxLayout()

        # Image
        image_label = QLabel()
        try:
            pixmap = QPixmap("indian_army_logo.png")  # Replace with your image path if needed
            pixmap = pixmap.scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
        except:
            image_label.setText("INDIAN ARMY\nOBJECT DETECTION")
            image_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #2e7d32;")
        image_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(image_label)

        layout.addLayout(header_layout)

        # Model configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(model_group)

        self.model_path_label = QLabel("No model loaded")
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_path_label)

        model_btn_layout = QHBoxLayout()
        self.browse_model_btn = QPushButton("Browse Model")
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setEnabled(False)
        model_btn_layout.addWidget(self.browse_model_btn)
        model_btn_layout.addWidget(self.load_model_btn)
        model_layout.addLayout(model_btn_layout)

        layout.addWidget(model_group)

        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)

        # Browse buttons layout
        browse_layout = QHBoxLayout()
        self.browse_btn = QPushButton("Browse Files/Folder")
        self.clear_files_btn = QPushButton("Clear Files")  # New clear button
        self.clear_files_btn.setEnabled(False)
        browse_layout.addWidget(self.browse_btn)
        browse_layout.addWidget(self.clear_files_btn)
        file_layout.addLayout(browse_layout)

        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous")
        self.next_btn = QPushButton("Next ▶")
        self.file_info_label = QLabel("No files loaded")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.file_info_label)
        nav_layout.addWidget(self.next_btn)
        file_layout.addLayout(nav_layout)

        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        layout.addWidget(file_group)

        # Detection controls
        detection_group = QGroupBox("Detection Controls")
        detection_layout = QVBoxLayout(detection_group)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(50)
        self.confidence_label = QLabel("0.50")
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        detection_layout.addLayout(conf_layout)

        # FPS control with clear labeling
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Playback FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 120)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setToolTip("Controls playback speed only. Output video saves at original FPS.")
        self.fps_display_label = QLabel("Current: 0.0")
        fps_layout.addWidget(self.fps_spinbox)
        fps_layout.addWidget(self.fps_display_label)
        detection_layout.addLayout(fps_layout)

        # Add note about FPS
        fps_note = QLabel(
            "📝 Note: Playback FPS affects viewing speed only.\nOutput video always saves at original FPS.")
        fps_note.setStyleSheet("font-size: 10px; color: #666; font-style: italic;")
        fps_note.setWordWrap(True)
        detection_layout.addWidget(fps_note)

        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)

        self.output_path_label = QLabel("No output folder selected")
        output_layout.addWidget(QLabel("Output Folder:"))
        output_layout.addWidget(self.output_path_label)

        output_btn_layout = QHBoxLayout()
        self.browse_output_btn = QPushButton("Browse Output")
        output_btn_layout.addWidget(self.browse_output_btn)
        output_layout.addLayout(output_btn_layout)

        # Auto-save video option
        self.auto_save_checkbox = QCheckBox("Auto-save detected videos")
        self.auto_save_checkbox.setChecked(True)
        self.auto_save_checkbox.setToolTip(
            "Videos are saved once at original FPS, regardless of playback speed or seeking")
        output_layout.addWidget(self.auto_save_checkbox)

        # NEW: Batch processing section for images
        batch_section = QGroupBox("Image Processing")
        batch_section.setStyleSheet("QGroupBox { color: #ff5722; font-weight: bold; }")
        batch_layout = QVBoxLayout(batch_section)

        # Save all images button
        self.save_all_images_btn = QPushButton("Save Images")
        self.save_all_images_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff5722;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #e64a19;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.save_all_images_btn.setEnabled(False)
        batch_layout.addWidget(self.save_all_images_btn)

        # Progress bar for batch processing
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_label = QLabel("")
        batch_layout.addWidget(self.batch_progress_bar)
        batch_layout.addWidget(self.batch_progress_label)

        output_layout.addWidget(batch_section)

        # Manual save options
        save_layout = QHBoxLayout()
        self.save_frame_btn = QPushButton("Save Current Frame")
        self.save_frame_btn.setEnabled(False)
        save_layout.addWidget(self.save_frame_btn)
        output_layout.addLayout(save_layout)

        # Save status label
        self.save_status_label = QLabel("")
        self.save_status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        output_layout.addWidget(self.save_status_label)

        layout.addWidget(output_group)

        # Play/Pause controls
        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.stop_btn)
        detection_layout.addLayout(control_layout)

        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        layout.addWidget(detection_group)

        layout.addStretch()
        return panel

    def create_display_panel(self):
        """Create right display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tab widget for original vs detected
        self.tab_widget = QTabWidget()

        # Original frame tab
        original_tab = QWidget()
        original_layout = QVBoxLayout(original_tab)

        # Zoom controls for original
        orig_zoom_layout = QHBoxLayout()
        self.orig_reset_zoom_btn = QPushButton("Reset View")
        orig_zoom_layout.addWidget(QLabel("Original Frame - Use mouse wheel to zoom"))
        orig_zoom_layout.addStretch()
        orig_zoom_layout.addWidget(self.orig_reset_zoom_btn)
        original_layout.addLayout(orig_zoom_layout)

        self.original_label = ZoomableLabel()
        original_layout.addWidget(self.original_label)
        self.tab_widget.addTab(original_tab, "Original")

        # Detected frame tab
        detected_tab = QWidget()
        detected_layout = QVBoxLayout(detected_tab)

        # Zoom controls for detected
        det_zoom_layout = QHBoxLayout()
        self.det_reset_zoom_btn = QPushButton("Reset View")
        det_zoom_layout.addWidget(QLabel("Detected Frame - Use mouse wheel to zoom"))
        det_zoom_layout.addStretch()
        det_zoom_layout.addWidget(self.det_reset_zoom_btn)
        detected_layout.addLayout(det_zoom_layout)

        # Create container for detected label and overlay
        container_widget = QWidget()
        container_layout = QVBoxLayout(container_widget)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Detected video frame
        self.detected_label = ZoomableLabel()
        self.detected_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container_layout.addWidget(self.detected_label)

        # Floating message label (overlay)
        self.overlay_label = QLabel("Frame saved")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 160);
            color: white;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 6px;
        """)
        self.overlay_label.setVisible(False)
        self.overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Add to layout
        container_layout.addWidget(self.overlay_label, alignment=Qt.AlignTop)

        # Add container to detected_layout
        detected_layout.addWidget(container_widget)
        self.tab_widget.addTab(detected_tab, "Detected")

        layout.addWidget(self.tab_widget)

        # Seek slider for video navigation
        seek_layout = QHBoxLayout()
        seek_layout.addWidget(QLabel("Video Position:"))
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)  # will be updated when video loads
        self.seek_slider.sliderReleased.connect(self.seek_video)
        self.seek_slider.setToolTip("Drag to seek through video. This won't affect saved output.")
        seek_layout.addWidget(self.seek_slider)
        layout.addLayout(seek_layout)

        return panel

    def setup_connections(self):
        """Setup signal-slot connections"""
        # Model loading
        self.browse_model_btn.clicked.connect(self.browse_model)
        self.load_model_btn.clicked.connect(self.load_model)

        # File selection
        self.browse_btn.clicked.connect(self.browse_files_or_folder)
        self.clear_files_btn.clicked.connect(self.clear_files)

        # Navigation
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)

        # Detection controls
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.fps_spinbox.valueChanged.connect(self.update_fps_real_time)
        self.play_btn.clicked.connect(self.start_detection)
        self.pause_btn.clicked.connect(self.pause_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        # Output
        self.browse_output_btn.clicked.connect(self.browse_output_folder)
        self.save_frame_btn.clicked.connect(self.save_current_frame)

        # NEW: Batch processing
        self.save_all_images_btn.clicked.connect(self.process_all_images)

        # Zoom controls
        self.orig_reset_zoom_btn.clicked.connect(self.original_label.reset_view)
        self.det_reset_zoom_btn.clicked.connect(self.detected_label.reset_view)

    def get_image_files_from_current_list(self):
        """Get all image files from current file list"""
        if not self.current_file_list:
            return []

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        return [f for f in self.current_file_list if Path(f).suffix.lower() in image_extensions]

    def process_all_images(self):
        """Process and save all images in the current file list"""
        if not self.detector:
            QMessageBox.warning(self, "No Model", "Please load a YOLO model first.")
            return

        image_files = self.get_image_files_from_current_list()
        if not image_files:
            QMessageBox.information(self, "No Images", "No image files found in the current file list.")
            return

        # Get output path
        output_path = self.output_path_label.toolTip()
        if not output_path or not os.path.exists(output_path):
            output_path = QFileDialog.getExistingDirectory(self, "Select Output Folder for Batch Processing")
            if not output_path:
                return
            self.output_path_label.setText(Path(output_path).name)
            self.output_path_label.setToolTip(output_path)

        # Confirm batch processing
        reply = QMessageBox.question(
            self,
            "Batch Process Images",
            f"Process and save {len(image_files)} images with object detection?\n\nThis will detect objects in all images and save them to the output folder.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Disable the button and show progress
        self.save_all_images_btn.setEnabled(False)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setMaximum(len(image_files))
        self.batch_progress_bar.setValue(0)
        self.batch_progress_label.setText(f"Processing 0/{len(image_files)} images...")

        # Start batch processing thread
        self.batch_processor = BatchImageProcessor(
            self.detector,
            image_files,
            output_path,
            self.save_queue
        )

        # Connect signals
        self.batch_processor.progress_update.connect(self.on_batch_progress)
        self.batch_processor.processing_complete.connect(self.on_batch_complete)
        self.batch_processor.error_occurred.connect(self.on_batch_error)

        self.batch_processor.start()

    @Slot(int, int)
    def on_batch_progress(self, current, total):
        """Update batch processing progress"""
        self.batch_progress_bar.setValue(current)
        self.batch_progress_label.setText(f"Processing {current}/{total} images...")

    @Slot(int)
    def on_batch_complete(self, processed_count):
        """Handle batch processing completion"""
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_label.setText("")
        self.save_all_images_btn.setEnabled(True)

        # Show completion message
        self.save_status_label.setText(f"✅ Batch processing complete! {processed_count} images queued for saving.")

        # Show success message box
        QMessageBox.information(
            self,
            "Batch Processing Complete",
            f"Successfully processed {processed_count} images!\n\nAll detected images have been queued for saving and will be saved in the background."
        )

        # Auto-hide status after 10 seconds
        QTimer.singleShot(10000, lambda: self.save_status_label.setText(""))

    @Slot(str)
    def on_batch_error(self, error_message):
        """Handle batch processing error"""
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_label.setText("")
        self.save_all_images_btn.setEnabled(True)

        QMessageBox.critical(self, "Batch Processing Error", error_message)

    def update_batch_button_state(self):
        """Update the state of the batch processing button"""
        image_files = self.get_image_files_from_current_list()
        has_images = len(image_files) > 0
        has_model = self.detector is not None

        self.save_all_images_btn.setEnabled(has_images and has_model)

        if has_images and has_model:
            self.save_all_images_btn.setText(f"🖼️ Process & Save All Images ({len(image_files)})")
        else:
            self.save_all_images_btn.setText("🖼️ Process & Save All Images")

    def clear_files(self):
        """Clear all selected files"""
        # Stop any running detection
        self.stop_detection()

        # Stop batch processing if running
        if self.batch_processor and self.batch_processor.isRunning():
            self.batch_processor.stop()

        # Clear file list
        self.current_file_list = []
        self.current_file_index = 0
        self.current_original_frame = None
        self.current_detected_frame = None

        # Clear displays
        self.original_label.clear()
        self.original_label.setText("No image loaded")
        self.detected_label.clear()
        self.detected_label.setText("No image loaded")

        # Reset seek slider
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setValue(0)

        # Clear save status
        self.save_status_label.setText("")

        # Hide batch progress
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_label.setText("")

        # Update UI
        self.update_navigation()
        self.update_batch_button_state()
        self.clear_files_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.play_btn.setText("▶ Play")
        self.save_frame_btn.setEnabled(False)

    def update_fps_real_time(self, value):
        """Update FPS in real-time for running videos"""
        # Update the detection thread if it's running
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.fps_changed.emit(value)

    def browse_model(self):
        """Browse for YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "PyTorch Models (*.pt)")
        if file_path:
            self.model_path_label.setText(Path(file_path).name)
            self.model_path_label.setToolTip(file_path)
            self.load_model_btn.setEnabled(True)

    def load_model(self):
        """Load the YOLO model"""
        model_path = self.model_path_label.toolTip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Error", "Model file not found")
            return

        try:
            self.detector = YOLODetector(model_path)
            self.model_path_label.setStyleSheet("color: #4caf50;")
            QMessageBox.information(self, "Success", f"Model loaded successfully: {Path(model_path).name}")
            self.enable_controls()
            self.update_batch_button_state()  # Update batch button state

            # Re-process current file with the new model
            if self.current_file_list:
                self.load_current_file()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")
            self.model_path_label.setStyleSheet("color: #f44336;")

    def seek_video(self):
        """Handle seek slider movement"""
        if (self.detection_thread and
                self.detection_thread.source_type == 'video' and
                self.detection_thread.cap):
            target_frame = self.seek_slider.value()
            self.detection_thread.seek_to_frame(target_frame)

    def browse_files_or_folder(self):
        """Browse for files or folder"""
        # Create dialog to choose between files and folder
        msg = QMessageBox()
        msg.setWindowTitle("Select Input Type")
        msg.setText("What would you like to select?")

        files_btn = msg.addButton("Select Files", QMessageBox.YesRole)
        folder_btn = msg.addButton("Select Folder", QMessageBox.NoRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)

        msg.exec()

        if msg.clickedButton() == files_btn:
            self.browse_files()
        elif msg.clickedButton() == folder_btn:
            self.browse_folder()

    def browse_files(self):
        """Browse for individual files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Media Files", "",
            "Media Files (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv)")
        if file_paths:
            self.load_files(file_paths)

    def browse_folder(self):
        """Browse for folder and load all supported files"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Find all supported files in the folder
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov', '.mkv']
            file_paths = []

            all_files = list(Path(folder_path).glob("*"))
            file_paths = [str(p) for p in all_files if p.suffix.lower() in supported_extensions]

            file_paths = [str(path) for path in file_paths]
            file_paths.sort()  # Sort files alphabetically

            if file_paths:
                self.load_files(file_paths)
            else:
                QMessageBox.information(self, "No Files", "No supported media files found in the selected folder.")

    def load_files(self, file_paths):
        """Load files into the application"""
        self.current_file_list = file_paths
        self.current_file_index = 0
        self.update_navigation()
        self.update_batch_button_state()  # Update batch button state
        self.clear_files_btn.setEnabled(True)  # Enable clear button

        self.load_current_file()

    def update_navigation(self):
        """Update navigation controls"""
        if not self.current_file_list:
            self.file_info_label.setText("No files loaded")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        current_file = self.current_file_list[self.current_file_index]
        filename = Path(current_file).name
        self.file_info_label.setText(f"{self.current_file_index + 1}/{len(self.current_file_list)}: {filename}")

        self.prev_btn.setEnabled(self.current_file_index > 0)
        self.next_btn.setEnabled(self.current_file_index < len(self.current_file_list) - 1)

    def show_previous(self):
        """Show previous file"""
        if self.current_file_index > 0:
            self.stop_detection()  # Stop any running detection
            self.current_file_index -= 1
            self.update_navigation()
            self.load_current_file()

    def show_next(self):
        """Show next file"""
        if self.current_file_index < len(self.current_file_list) - 1:
            self.stop_detection()  # Stop any running detection
            self.current_file_index += 1
            self.update_navigation()
            self.load_current_file()

    def load_current_file(self):
        """Load and display current file"""
        if not self.current_file_list:
            return

        # Clear previous save status
        self.save_status_label.setText("")

        current_file = self.current_file_list[self.current_file_index]
        file_ext = Path(current_file).suffix.lower()

        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Load image
            frame = cv2.imread(current_file)
            if frame is not None:
                # Always show original, detect only if model is loaded
                if self.detector:
                    detected_frame = self.detector.detect(frame)
                else:
                    detected_frame = frame.copy()  # Show original as detected if no model
                self.display_frames(frame, detected_frame)
                self.save_frame_btn.setEnabled(True)

        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # For video, show first frame and setup video controls
            cap = cv2.VideoCapture(current_file)
            ret, frame = cap.read()

            if ret:
                # Setup seek slider for video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.seek_slider.setRange(0, total_frames - 1)
                self.seek_slider.setValue(0)

                # Always show original, detect only if model is loaded
                if self.detector:
                    detected_frame = self.detector.detect(frame)
                else:
                    detected_frame = frame.copy()  # Show original as detected if no model
                self.display_frames(frame, detected_frame)
                self.enable_video_controls()
                self.save_frame_btn.setEnabled(True)

            cap.release()

    def enable_controls(self):
        """Enable controls when model is loaded"""
        # Enable play button regardless of model if we have files
        if self.current_file_list:
            current_file = self.current_file_list[self.current_file_index]
            file_ext = Path(current_file).suffix.lower()
            is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']
            self.play_btn.setEnabled(is_video)
        self.save_frame_btn.setEnabled(bool(self.current_file_list))

    def enable_video_controls(self):
        """Enable video-specific controls"""
        if self.current_file_list:
            current_file = self.current_file_list[self.current_file_index]
            file_ext = Path(current_file).suffix.lower()
            is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']
            self.play_btn.setEnabled(is_video)

    def update_confidence(self, value):
        """Update confidence threshold"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")

        if self.detector:
            self.detector.set_confidence(confidence)

            # 🔄 If current file is an image, reprocess immediately
            if self.current_file_list:
                current_file = self.current_file_list[self.current_file_index]
                file_ext = Path(current_file).suffix.lower()
                if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    frame = cv2.imread(current_file)
                    if frame is not None:
                        detected_frame = self.detector.detect(frame)
                        self.display_frames(frame, detected_frame)

    def start_detection(self):
        """Start or resume detection process with auto-save for videos only"""
        if not self.current_file_list:
            QMessageBox.warning(self, "Error", "No files loaded")
            return

        # If thread exists and is paused, just resume
        if (self.detection_thread and
                self.detection_thread.isRunning() and
                self.detection_thread.is_paused):
            self.detection_thread.resume()
            self.play_btn.setText("▶ Play")
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            return

        current_file = self.current_file_list[self.current_file_index]

        # Check if we need output path for saving
        output_path = self.output_path_label.toolTip()
        save_video = False

        # Only auto-save videos, not individual images (use batch processing for images)
        file_ext = Path(current_file).suffix.lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']

        if is_video and self.detector and output_path and os.path.exists(output_path):
            if self.auto_save_checkbox.isChecked():
                save_video = True
        elif is_video and self.detector and not output_path:
            # Ask user if they want to select output folder for video
            reply = QMessageBox.question(self, "No Output Folder",
                                         "No output folder selected. Would you like to select one to save detected video?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
                if folder_path:
                    self.output_path_label.setText(Path(folder_path).name)
                    self.output_path_label.setToolTip(folder_path)
                    output_path = folder_path
                    if self.auto_save_checkbox.isChecked():
                        save_video = True

        # Stop any existing thread
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()

        # Clear previous save status
        self.save_status_label.setText("")

        # Create and start detection thread (no auto-save for individual images)
        playback_fps = self.fps_spinbox.value()
        self.detection_thread = DetectionThread(
            self.detector,
            current_file,
            playback_fps,
            save_video,  # Only for videos
            output_path,
            self.save_queue,
            False  # Don't auto-save individual images - use batch processing instead
        )

        # Connect signals
        self.detection_thread.frame_ready.connect(self.on_frame_ready)
        self.detection_thread.fps_update.connect(self.on_fps_update)
        self.detection_thread.error_occurred.connect(self.on_error)
        self.detection_thread.saving_completed.connect(self.on_saving_completed)

        self.detection_thread.start()

        # Update UI
        self.play_btn.setText("▶ Play")
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        # Show save status for videos only
        if save_video:
            self.save_status_label.setText("🔴 Saving detected video...")

    def pause_detection(self):
        """Pause detection process"""
        if self.detection_thread:
            self.detection_thread.pause()
            # Update UI to show resume option
            self.play_btn.setText("▶ Resume")
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)

    def stop_detection(self):
        """Stop detection process"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread = None

        # Update UI
        self.play_btn.setText("▶ Play")
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.fps_display_label.setText("Current: 0.0")

    def browse_output_folder(self):
        """Browse for output folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_path_label.setText(Path(folder_path).name)
            self.output_path_label.setToolTip(folder_path)

    def save_current_frame(self):
        """Save current detected frame and show overlay message"""
        if self.current_detected_frame is None:
            return

        output_folder = self.output_path_label.toolTip()
        if not output_folder:
            output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder for Frame")
            if not output_folder:
                return
            self.output_path_label.setText(Path(output_folder).name)
            self.output_path_label.setToolTip(output_folder)

        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            current_file = self.current_file_list[self.current_file_index]
            base_name = Path(current_file).stem
            filename = f"{base_name}_detected_{timestamp}.jpg"

            # Add to save queue
            self.save_queue.add_to_queue('image', self.current_detected_frame, output_folder, filename)

            # ✅ Show overlay message (toast-style)
            self.overlay_label.setText("✅ Frame queued for saving")
            self.overlay_label.setVisible(True)

            # Auto-hide after 3 seconds
            QTimer.singleShot(3000, lambda: self.overlay_label.setVisible(False))

        except Exception as e:
            self.overlay_label.setText("❌ Error queuing frame")
            self.overlay_label.setVisible(True)
            QTimer.singleShot(3000, lambda: self.overlay_label.setVisible(False))

    @Slot(np.ndarray, np.ndarray)
    def on_frame_ready(self, original_frame, detected_frame):
        """Handle new frame from detection thread"""
        self.current_original_frame = original_frame
        self.current_detected_frame = detected_frame
        self.display_frames(original_frame, detected_frame)
        self.save_frame_btn.setEnabled(True)

        # Update seek slider position for videos
        if (self.detection_thread and
                self.detection_thread.source_type == 'video' and
                self.detection_thread.cap):

            current_frame = self.detection_thread.current_frame_index
            # Only update if not currently being dragged by user
            if not self.seek_slider.isSliderDown():
                self.seek_slider.setValue(current_frame)

    @Slot(float)
    def on_fps_update(self, fps):
        """Update FPS display"""
        self.fps_display_label.setText(f"Current: {fps:.1f}")

    @Slot(str)
    def on_saving_completed(self, message):
        """Handle video saving completion"""
        # Update save status to show completion
        self.save_status_label.setText(f"✅ {message}")
        self.save_status_label.setStyleSheet("color: #4caf50; font-weight: bold;")

    def on_error(self, error_message):
        """Handle error from detection thread"""
        QMessageBox.critical(self, "Detection Error", error_message)
        self.stop_detection()

    def display_frames(self, original_frame, detected_frame):
        """Display original and detected frames"""
        self.original_label.set_image(original_frame)
        self.detected_label.set_image(detected_frame)

    def closeEvent(self, event):
        """Handle application close"""
        # Stop detection thread
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()

        # Stop batch processing thread
        if self.batch_processor and self.batch_processor.isRunning():
            self.batch_processor.stop()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #2196f3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1976d2;
        }
        QPushButton:pressed {
            background-color: #0d47a1;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QSpinBox, QSlider {
            padding: 4px;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())