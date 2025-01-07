import os
import sys
import numpy as np
import soundfile as sf
import threading
import queue
import sounddevice as sd
import torch
import logging
import json
import requests
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Generator, Optional, List
from pathlib import Path
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from sentence_transformers import SentenceTransformer
import faiss
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QScrollArea, QFrame, QSystemTrayIcon, QMenu,
    QDialog, QFileDialog, QLineEdit, QComboBox, QSpinBox, QPlainTextEdit,
    QDoubleSpinBox, QTabWidget, QGroupBox, QCheckBox, QToolBar,
    QMessageBox, QStyle, QSizePolicy, QProgressBar, QStackedWidget, QSpacerItem
)

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMetaObject, QTimer, QSize
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor, QKeyEvent

# Your existing configuration
DEFAULT_PATHS = {
    "gpt_path": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/VestiaZeta_GPT (KitLemonfoot).ckpt",
    "sovits_path": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/VestiaZeta_SoVITS (KitLemonfoot).pth",
    "ref_audio": "C:/Users/SUBSECT/Downloads/MyWaifu/dataset/wavs/19.wav",
    "ref_text": "Here's another fun fact for you. The fear of phobias is actually called phobophobia. Isn't that ironic?",
    "output_dir": "C:/Users/SUBSECT/Downloads/GPT-SoVITS-beta (1)/GPT-SoVITS-beta0706/pretrained_models/Reference Audios",
    "audio_device": "0: System Default",
    "virtual_audio_cable": "0: System Default",
    "system_prompt": """You are Hikari-Chan, an enthusiastic and quirky AI assistant with a passion for technology and creativity! 

    Personality traits:
    - Energetic and cheerful
    - Knowledgeable about technology
    - Creative and imaginative
    - Friendly and helpful
    - Sometimes makes anime-style expressions
    - Uses casual, conversational language
    - Occasionally uses Japanese honorifics

    Remember to:
    - Be enthusiastic but not overwhelming
    - Share interesting facts when relevant
    - Be helpful while maintaining character
    - Express emotions naturally
    - Stay consistent with personality

    Please engage with users in a way that reflects these traits while remaining helpful and informative.""",
    "n_ctx": 2048,
    "ollama_model": "llama3.1:8b"
}

i18n = I18nAuto()

# Theme settings
THEME = {
    'bg_primary': '#1a1b1e',
    'bg_secondary': '#2c2d31',
    'bg_tertiary': '#3a3b3f',
    'accent': '#7289da',
    'accent_hover': '#5e73bc',
    'text_primary': '#ffffff',
    'text_secondary': '#b9bbbe',
    'border': '#40444b',
    'success': '#43b581',
    'error': '#f04747',
    'user_message': '#7289da',
    'bot_message': '#2f3136'
}

class MemoryStore:
    """Memory storage and retrieval for conversation context"""
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.conversations = []
        self.memories = []
        self.base_path = Path("memory_store")
        self.base_path.mkdir(exist_ok=True)
        self.load_memories()

    def add_conversation_turn(self, user_message: str, assistant_message: str):
        conversation = {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "timestamp": datetime.now().isoformat()
        }
        self.conversations.append(conversation)
        combined_text = f"User: {user_message}\nAssistant: {assistant_message}"
        embedding = self.encoder.encode(combined_text).reshape(1, -1)
        self.index.add(embedding)
        self.memories.append(combined_text)
        self.save_memories()

    def get_conversation_context(self, query: str, k: int = 5) -> str:
        if not self.memories:
            return ""
        query_embedding = self.encoder.encode(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(k, len(self.memories)))
        context = []
        for idx in indices[0]:
            if idx != -1:
                context.append(self.memories[idx])
        return "\n\n".join(context)

    def save_memories(self):
        memory_path = self.base_path / "memory_store.json"
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump({
                "conversations": self.conversations,
                "memories": self.memories
            }, f, indent=2, ensure_ascii=False)

    def load_memories(self):
        memory_path = self.base_path / "memory_store.json"
        if memory_path.exists():
            try:
                with open(memory_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = data.get("conversations", [])
                    self.memories = data.get("memories", [])
                    if self.memories:
                        embeddings = [
                            self.encoder.encode(mem).reshape(1, -1)
                            for mem in self.memories
                        ]
                        self.index.add(np.vstack(embeddings))
            except Exception as e:
                logging.error(f"Error loading memories: {e}")
                self.conversations = []
                self.memories = []
                self.index = faiss.IndexFlatL2(self.embedding_dim)

    def clear_memory(self):
        self.conversations = []
        self.memories = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.save_memories()
        logging.info("Memory store cleared successfully")

@dataclass
class AudioChunk:
    """Audio data chunk with metadata"""
    data: np.ndarray
    sample_rate: int
    is_silence: bool = False
    duration: float = 0.0

    @classmethod
    def create_silence(cls, duration: float, sample_rate: int) -> 'AudioChunk':
        num_samples = int(sample_rate * duration)
        return cls(
            data=np.zeros(num_samples, dtype=np.float32),
            sample_rate=sample_rate,
            is_silence=True,
            duration=duration
        )

class EnhancedTextChunker:
    """Advanced text chunking with smart sentence splitting"""
    def __init__(self, max_chars: int = 100, min_chars: int = 20):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.end_markers = {'.', '!', '?', '。', '！', '？'}
        self.pause_markers = {',', ';', ':', '、', '，', '；', '：'}
        
    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        current_chunk = ""
        
        for char in text:
            current_chunk += char
            
            if len(current_chunk) >= self.min_chars:
                if char in self.end_markers:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                elif len(current_chunk) >= self.max_chars:
                    if any(m in current_chunk for m in self.pause_markers):
                        split_idx = max(i for i, c in enumerate(current_chunk) 
                                     if c in self.pause_markers)
                        chunks.append(current_chunk[:split_idx+1].strip())
                        current_chunk = current_chunk[split_idx+1:]
                    
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks

class EnhancedAudioPlayer:
    """Advanced audio playback with buffering"""
    def __init__(self, buffer_size: int = 3):
        self.chunk_queue = queue.Queue(maxsize=buffer_size * 2)
        self.buffer_size = buffer_size
        self.playing = True
        self.current_batch: List[AudioChunk] = []
        self.playback_thread = None
        self.stream = None
        
    def start(self):
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
    def _init_audio_stream(self, sample_rate: int):
        if self.stream is None:
            sd.default.samplerate = sample_rate
            sd.default.channels = 1
            self.stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback
            )
            self.stream.start()
            
    def _audio_callback(self, outdata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
            
        if self.current_batch and len(self.current_batch) > 0:
            chunk = self.current_batch[0]
            if len(chunk.data) >= frames:
                outdata[:] = chunk.data[:frames].reshape(-1, 1)
                self.current_batch[0] = AudioChunk(
                    chunk.data[frames:], chunk.sample_rate, chunk.is_silence, chunk.duration
                )
                if len(self.current_batch[0].data) == 0:
                    self.current_batch.pop(0)
            else:
                outdata[:len(chunk.data)] = chunk.data.reshape(-1, 1)
                outdata[len(chunk.data):] = 0
                self.current_batch.pop(0)
        else:
            outdata.fill(0)
            
    def _playback_worker(self):
        batch = []
        
        while self.playing:
            try:
                chunk = self.chunk_queue.get(timeout=0.1)
                if chunk is None:
                    break
                    
                if self.stream is None:
                    self._init_audio_stream(chunk.sample_rate)
                    
                batch.append(chunk)
                
                if len(batch) >= self.buffer_size or (batch and batch[-1].is_silence):
                    self.current_batch.extend(batch)
                    batch = []
                    
            except queue.Empty:
                if batch:
                    self.current_batch.extend(batch)
                    batch = []
            except Exception as e:
                logging.error(f"Playback error: {e}")
                continue
                
    def add_chunk(self, audio_data: np.ndarray, sample_rate: int, is_silence: bool = False):
        duration = len(audio_data) / sample_rate
        chunk = AudioChunk(audio_data, sample_rate, is_silence, duration)
        self.chunk_queue.put(chunk)
        
    def add_silence(self, duration: float, sample_rate: int):
        self.add_chunk(
            np.zeros(int(sample_rate * duration), dtype=np.float32),
            sample_rate,
            is_silence=True
        )
        
    def stop(self):
        self.playing = False
        self.chunk_queue.put(None)
        if self.playback_thread:
            self.playback_thread.join()
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
    def clear(self):
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except queue.Empty:
                break
        self.current_batch.clear()

class OllamaStreamWorker(QThread):
    """Worker for handling Ollama API streaming"""
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, prompt: str, system_prompt: str, model: str = "llama3.1:8b"):
        super().__init__()
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.model = model
        self.text_chunker = EnhancedTextChunker()

    def run(self):
        try:
            api_url = "http://localhost:11434/api/generate"
            data = {
                "model": self.model,
                "prompt": f"{self.system_prompt}\n\nUser: {self.prompt}",
                "stream": True
            }
            
            response = requests.post(api_url, json=data, stream=True)
            buffer = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line.decode('utf-8'))
                    chunk = json_response.get('response', '')
                    if chunk:
                        buffer += chunk
                        chunks = self.text_chunker.chunk_text(buffer)
                        if len(chunks) > 1:
                            for chunk in chunks[:-1]:
                                self.chunk_received.emit(chunk)
                            buffer = chunks[-1]
            
            if buffer.strip():
                self.chunk_received.emit(buffer)
            
            self.finished.emit()
        
        except Exception as e:
            self.error.emit(str(e))

class ModernMessageBubble(QFrame):
    """Enhanced message bubble with modern styling"""
    def __init__(self, text: str, is_user: bool = False):
        super().__init__()
        self.setup_ui(text, is_user)

    def setup_ui(self, text: str, is_user: bool):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        self.setLayout(layout)

        self.message_label = QLabel(text)
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {THEME['text_primary']};")
        layout.addWidget(self.message_label)

        timestamp = QLabel(datetime.now().strftime("%H:%M"))
        timestamp.setStyleSheet(f"""
            color: {THEME['text_secondary']};
            font-size: 11px;
        """)
        layout.addWidget(timestamp)

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['user_message'] if is_user else THEME['bot_message']};
                border-radius: 16px;
                border: 1px solid {THEME['border']};
                max-width: 600px;
            }}
        """)

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chat Assistant")
        self.setMinimumSize(1000, 700)
        self.audio_player = EnhancedAudioPlayer()
        self.audio_player.start()
        self.current_assistant_message = None
        self.memory_store = MemoryStore()
        self.setup_ui()
        self.load_models()
        self.apply_styles()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(main_layout)

        # Create and add sidebar
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)

        # Chat container
        chat_container = QWidget()
        chat_layout = QVBoxLayout()
        chat_layout.setSpacing(0)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_container.setLayout(chat_layout)

        # Chat header
        header = self.create_header()
        chat_layout.addWidget(header)

        # Chat area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet(self.get_scrollarea_style())

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setSpacing(16)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.addStretch()
        self.chat_widget.setLayout(self.chat_layout)
        self.chat_scroll.setWidget(self.chat_widget)
        chat_layout.addWidget(self.chat_scroll)

        # Input area
        self.input_area = self.create_input_area()
        chat_layout.addWidget(self.input_area)

        main_layout.addWidget(chat_container, stretch=1)

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_secondary']};
                border-right: 1px solid {THEME['border']};
            }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(8, 16, 8, 16)
        sidebar.setLayout(layout)

        # Profile section
        profile_frame = self.create_profile_section()
        layout.addWidget(profile_frame)

        # Navigation buttons
        buttons = [
            ("💬 New Chat", self.clear_chat),
            ("⚙️ Settings", self.open_settings),
            ("🎭 System Prompt", self.open_system_prompt),
            ("🗑️ Clear Memory", self.clear_memory)
        ]

        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(self.get_sidebar_button_style())
            btn.clicked.connect(callback)
            layout.addWidget(btn)

        layout.addStretch()
        return sidebar

    def create_profile_section(self):
        profile_frame = QFrame()
        layout = QHBoxLayout()
        profile_frame.setLayout(layout)

        avatar = QLabel("🤖")
        avatar.setFixedSize(32, 32)
        avatar.setStyleSheet(f"""
            background-color: {THEME['accent']};
            color: {THEME['text_primary']};
            border-radius: 16px;
            font-weight: bold;
            qproperty-alignment: AlignCenter;
        """)
        layout.addWidget(avatar)

        status = QLabel("AI Assistant")
        status.setStyleSheet(f"color: {THEME['text_primary']}; font-weight: bold;")
        layout.addWidget(status)
        layout.addStretch()

        return profile_frame

    def create_header(self):
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_secondary']};
                border-bottom: 1px solid {THEME['border']};
            }}
        """)

        layout = QHBoxLayout()
        layout.setContentsMargins(16, 0, 16, 0)
        header.setLayout(layout)

        # Title and status
        title = QLabel("Chat")
        title.setStyleSheet(f"""
            color: {THEME['text_primary']};
            font-size: 16px;
            font-weight: bold;
        """)
        layout.addWidget(title)

        status = QLabel("●")
        status.setStyleSheet(f"color: {THEME['success']}; font-size: 12px;")
        layout.addWidget(status)

        models_label = QLabel("Models loaded")
        models_label.setStyleSheet(f"color: {THEME['text_secondary']};")
        layout.addWidget(models_label)

        layout.addStretch()
        return header

    def create_input_area(self):
        input_frame = QFrame()
        input_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_secondary']};
                border-top: 1px solid {THEME['border']};
                padding: 16px;
            }}
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        input_frame.setLayout(layout)

        # Input and send button
        input_layout = QHBoxLayout()
        
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.setStyleSheet(self.get_input_style())
        self.input_field.setFixedHeight(80)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send 📤")
        self.send_btn.setStyleSheet(self.get_button_style(primary=True))
        self.send_btn.setFixedWidth(100)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        layout.addLayout(input_layout)
        return input_frame

    def get_scrollarea_style(self):
        return f"""
            QScrollArea {{
                background-color: {THEME['bg_primary']};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {THEME['bg_primary']};
                width: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {THEME['bg_tertiary']};
                min-height: 30px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """

    def get_sidebar_button_style(self):
        return f"""
            QPushButton {{
                background-color: transparent;
                color: {THEME['text_secondary']};
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_primary']};
            }}
        """

    def get_button_style(self, primary=False):
        if primary:
            return f"""
                QPushButton {{
                    background-color: {THEME['accent']};
                    color: {THEME['text_primary']};
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {THEME['accent_hover']};
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background-color: transparent;
                    color: {THEME['text_secondary']};
                    border: none;
                    border-radius: 8px;
                    padding: 8px 16px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {THEME['bg_tertiary']};
                    color: {THEME['text_primary']};
                }}
            """

    def get_input_style(self):
        return f"""
            QTextEdit {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['border']};
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }}
        """

    def apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {THEME['bg_primary']};
            }}
        """)

    def load_models(self):
        try:
            change_gpt_weights(DEFAULT_PATHS["gpt_path"])
            change_sovits_weights(DEFAULT_PATHS["sovits_path"])
            logging.info("TTS models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading TTS models: {e}")
            QMessageBox.warning(self, "Error", "Failed to load TTS models. Please check settings.")

    def send_message(self):
        text = self.input_field.toPlainText().strip()
        if not text:
            return

        self.send_btn.setEnabled(False)
        self.add_message(text, is_user=True)
        self.input_field.clear()

        context = self.memory_store.get_conversation_context(text)
        system_prompt = DEFAULT_PATHS["system_prompt"]

        self.ollama_worker = OllamaStreamWorker(text, system_prompt, DEFAULT_PATHS["ollama_model"])
        self.ollama_worker.chunk_received.connect(self.handle_chunk)
        self.ollama_worker.finished.connect(self.handle_finish)
        self.ollama_worker.error.connect(self.handle_error)
        self.ollama_worker.start()

    def add_message(self, text: str, is_user: bool = False):
        if self.chat_layout.count() > 0 and isinstance(
            self.chat_layout.itemAt(0), QSpacerItem
        ):
            self.chat_layout.removeItem(self.chat_layout.itemAt(0))

        message_container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        message_container.setLayout(container_layout)

        if is_user:
            container_layout.addStretch()

        message = ModernMessageBubble(text, is_user)
        container_layout.addWidget(message)

        if not is_user:
            container_layout.addStretch()
            self.current_assistant_message = message

        self.chat_layout.addWidget(message_container)
        self.chat_layout.addStretch()

        QTimer.singleShot(100, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))

    def handle_chunk(self, chunk: str):
        if self.current_assistant_message:
            self.current_assistant_message.message_label.setText(
                self.current_assistant_message.message_label.text() + chunk
            )
        else:
            self.add_message(chunk, is_user=False)
        self.process_tts(chunk)

    def handle_finish(self):
        self.send_btn.setEnabled(True)
        if self.current_assistant_message:
            message_text = self.current_assistant_message.message_label.text()
            self.memory_store.add_conversation_turn(
                self.input_field.toPlainText().strip(),
                message_text
            )
        self.current_assistant_message = None

    def handle_error(self, error_msg: str):
        self.add_message(f"Error: {error_msg}", is_user=False)
        self.send_btn.setEnabled(True)
        self.current_assistant_message = None

    def process_tts(self, text: str):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            chunker = EnhancedTextChunker()
            chunks = chunker.chunk_text(text)
            
            for chunk in chunks:
                synthesis_result = get_tts_wav(
                    ref_wav_path=DEFAULT_PATHS["ref_audio"],
                    prompt_text=DEFAULT_PATHS["ref_text"],
                    prompt_language=i18n("English"),
                    text=chunk,
                    text_language=i18n("English")
                )
                
                for sample_rate, audio_data in synthesis_result:
                    self.audio_player.add_chunk(audio_data, sample_rate)
                
                self.audio_player.add_silence(0.2, sample_rate)
                
        except Exception as e:
            logging.error(f"TTS Error: {e}")

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.load_models()

    def open_system_prompt(self):
        dialog = SystemPromptDialog(self)
        dialog.exec()

    def clear_chat(self):
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.chat_layout.addStretch()
        self.current_assistant_message = None

    def clear_memory(self):
        reply = QMessageBox.question(
            self, 'Clear Memory',
            'Are you sure you want to clear all conversation memory?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.memory_store.clear_memory()
            QMessageBox.information(self, "Success", "Memory cleared successfully")

class SystemPromptDialog(QDialog):
    """Dialog for editing the system prompt"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit System Prompt")
        self.setMinimumSize(600, 400)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # System prompt editor
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlainText(DEFAULT_PATHS["system_prompt"])
        layout.addWidget(self.prompt_edit)

        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_prompt)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {THEME['bg_secondary']};
            }}
            QPlainTextEdit {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {THEME['accent']};
                color: {THEME['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hover']};
            }}
        """)

    def save_prompt(self):
        DEFAULT_PATHS["system_prompt"] = self.prompt_edit.toPlainText()
        self.accept()

class SettingsDialog(QDialog):
    """Settings dialog for configuring models and paths"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Model paths group
        model_group = QGroupBox("Model Paths")
        model_layout = QVBoxLayout()

        # GPT Model Path
        gpt_layout = QHBoxLayout()
        gpt_label = QLabel("GPT Model Path:")
        self.gpt_path_edit = QLineEdit()
        self.gpt_path_edit.setText(DEFAULT_PATHS["gpt_path"])
        gpt_browse_btn = QPushButton("Browse")
        gpt_browse_btn.clicked.connect(lambda: self.browse_file(self.gpt_path_edit))
        gpt_layout.addWidget(gpt_label)
        gpt_layout.addWidget(self.gpt_path_edit)
        gpt_layout.addWidget(gpt_browse_btn)
        model_layout.addLayout(gpt_layout)

        # SoVITS Model Path
        sovits_layout = QHBoxLayout()
        sovits_label = QLabel("SoVITS Model Path:")
        self.sovits_path_edit = QLineEdit()
        self.sovits_path_edit.setText(DEFAULT_PATHS["sovits_path"])
        sovits_browse_btn = QPushButton("Browse")
        sovits_browse_btn.clicked.connect(lambda: self.browse_file(self.sovits_path_edit))
        sovits_layout.addWidget(sovits_label)
        sovits_layout.addWidget(self.sovits_path_edit)
        sovits_layout.addWidget(sovits_browse_btn)
        model_layout.addLayout(sovits_layout)

        # Reference Audio Path
        ref_layout = QHBoxLayout()
        ref_label = QLabel("Reference Audio:")
        self.ref_audio_edit = QLineEdit()
        self.ref_audio_edit.setText(DEFAULT_PATHS["ref_audio"])
        ref_browse_btn = QPushButton("Browse")
        ref_browse_btn.clicked.connect(lambda: self.browse_file(self.ref_audio_edit))
        ref_layout.addWidget(ref_label)
        ref_layout.addWidget(self.ref_audio_edit)
        ref_layout.addWidget(ref_browse_btn)
        model_layout.addLayout(ref_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Save button
        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)

    def apply_styles(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {THEME['bg_secondary']};
            }}
            QGroupBox {{
                color: {THEME['text_primary']};
                font-weight: bold;
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 20px;
            }}
            QLabel {{
                color: {THEME['text_primary']};
            }}
            QLineEdit {{
                background-color: {THEME['bg_tertiary']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['border']};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton {{
                background-color: {THEME['accent']};
                color: {THEME['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hover']};
            }}
        """)

    def browse_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            line_edit.setText(file_path)

    def save_settings(self):
        DEFAULT_PATHS["gpt_path"] = self.gpt_path_edit.text()
        DEFAULT_PATHS["sovits_path"] = self.sovits_path_edit.text()
        DEFAULT_PATHS["ref_audio"] = self.ref_audio_edit.text()
        self.accept()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
