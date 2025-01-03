import tkinter as tk
from tkinter import ttk, messagebox
import vlc
import time
import csv
import yt_dlp
import os
import threading
import wave
import pyaudio
from pathlib import Path
import json

class EnhancedTimestampTool:
    def __init__(self, root):
        self.root = root
        self.root.title("2B Voiceline Collector")
        self.root.geometry("1200x800")
        
        # State variables
        self.start_time = 0
        self.end_time = 0
        self.current_url = ""
        self.clips = []
        self.load_saved_clips()
        
        self.setup_ui()
        self.setup_shortcuts()
        
    def setup_ui(self):
        # Main layout with left panel and right panel
        self.paned = ttk.PanedWindow(self.root, orient='horizontal')
        self.paned.pack(fill='both', expand=True)
        
        # Left panel for video/controls
        self.left_panel = ttk.Frame(self.paned)
        self.paned.add(self.left_panel, weight=2)
        
        # URL Entry with load button
        self.url_frame = ttk.Frame(self.left_panel)
        self.url_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(self.url_frame, text="YouTube URL:").pack(side='left')
        self.url_entry = ttk.Entry(self.url_frame)
        self.url_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(self.url_frame, text="Load", command=self.load_video).pack(side='left')
        
        # Video frame
        self.video_frame = ttk.Frame(self.left_panel, relief='sunken', borderwidth=1)
        self.video_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Controls under video
        self.control_frame = ttk.Frame(self.left_panel)
        self.control_frame.pack(fill='x', padx=5, pady=5)
        
        # Time slider
        self.time_slider = ttk.Scale(self.control_frame, from_=0, to=100, orient='horizontal')
        self.time_slider.pack(fill='x', padx=5)
        
        # Control buttons
        self.button_frame = ttk.Frame(self.control_frame)
        self.button_frame.pack(fill='x')
        
        ttk.Button(self.button_frame, text="⏮ -5s", command=lambda: self.seek_relative(-5)).pack(side='left')
        self.play_button = ttk.Button(self.button_frame, text="▶", command=self.toggle_play)
        self.play_button.pack(side='left')
        ttk.Button(self.button_frame, text="⏭ +5s", command=lambda: self.seek_relative(5)).pack(side='left')
        
        # Timestamp controls
        self.time_frame = ttk.Frame(self.control_frame)
        self.time_frame.pack(fill='x')
        
        ttk.Button(self.time_frame, text="Mark Start", command=self.mark_start).pack(side='left')
        ttk.Button(self.time_frame, text="Mark End", command=self.mark_end).pack(side='left')
        self.time_label = ttk.Label(self.time_frame, text="00:00 / 00:00")
        self.time_label.pack(side='left', padx=10)
        
        # Clip info
        self.info_frame = ttk.Frame(self.left_panel)
        self.info_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(self.info_frame, text="Transcript:").pack(side='left')
        self.transcript_entry = ttk.Entry(self.info_frame)
        self.transcript_entry.pack(side='left', fill='x', expand=True)
        
        ttk.Label(self.info_frame, text="Context:").pack(side='left')
        self.context_entry = ttk.Entry(self.info_frame)
        self.context_entry.pack(side='left', fill='x', expand=True)
        
        ttk.Button(self.info_frame, text="Save Clip", command=self.save_clip).pack(side='left')
        
        # Right panel for clip list
        self.right_panel = ttk.Frame(self.paned)
        self.paned.add(self.right_panel, weight=1)
        
        # Clip list with scrollbar
        self.clip_frame = ttk.Frame(self.right_panel)
        self.clip_frame.pack(fill='both', expand=True)
        
        self.clip_tree = ttk.Treeview(self.clip_frame, columns=('Time', 'Text'), show='headings')
        self.clip_tree.heading('Time', text='Time')
        self.clip_tree.heading('Text', text='Text')
        self.clip_tree.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(self.clip_frame, orient='vertical', command=self.clip_tree.yview)
        scrollbar.pack(side='right', fill='y')
        self.clip_tree.configure(yscrollcommand=scrollbar.set)
        
        # Clip controls
        self.clip_controls = ttk.Frame(self.right_panel)
        self.clip_controls.pack(fill='x')
        
        ttk.Button(self.clip_controls, text="Play Clip", command=self.play_selected_clip).pack(side='left')
        ttk.Button(self.clip_controls, text="Delete", command=self.delete_selected_clip).pack(side='left')
        ttk.Button(self.clip_controls, text="Export All", command=self.export_clips).pack(side='left')
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
        self.status_bar.pack(fill='x', side='bottom')

    def setup_shortcuts(self):
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Left>', lambda e: self.seek_relative(-5))
        self.root.bind('<Right>', lambda e: self.seek_relative(5))
        self.root.bind('<Control-s>', lambda e: self.save_clip())
        
    def load_video(self):
        url = self.url_entry.get()
        if not url:
            return
                
        self.status_var.set("Loading video...")
        
        def handle_error(error_msg):
            self.status_var.set(f"Error: {error_msg}")
            messagebox.showerror("Error", f"Failed to load video: {error_msg}")
        
        def download_thread():
            try:
                # Configure yt-dlp options
                ydl_opts = {
                    'format': 'bestvideo+bestaudio/best',
                    'quiet': True,
                    'no_warnings': True
                }
                
                # Extract video information
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_url = info['url']
                    
                def init():
                    self.initialize_player(video_url)
                    
                # Schedule player initialization in main thread
                self.root.after(0, init)
                
            except Exception as e:
                # Schedule error handling in main thread with captured error
                error_msg = str(e)
                self.root.after(0, lambda: handle_error(error_msg))
                    
        # Start download thread
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

    def initialize_player(self, url):
        try:
            # Initialize VLC instance and player
            if not hasattr(self, 'instance'):
                self.instance = vlc.Instance()
            if not hasattr(self, 'player'):
                self.player = self.instance.media_player_new()
                
            # Create and set media
            self.media = self.instance.media_new(url)
            self.player.set_media(self.media)
            
            # Configure video output
            if os.name == 'nt':  # Windows
                self.player.set_hwnd(self.video_frame.winfo_id())
            else:  # Linux/Mac
                self.player.set_xwindow(self.video_frame.winfo_id())
                
            # Start playback and updates
            self.player.play()
            self.status_var.set("Video loaded successfully")
            self.update_time()
            
        except Exception as error:
            self.status_var.set(f"Player error: {str(error)}")
            messagebox.showerror("Error", f"Failed to initialize player: {str(error)}")
        
    def update_time(self):
        if not hasattr(self, 'player'):
            return
            
        # Update time display and slider
        if self.player.is_playing():
            time_str = time.strftime('%M:%S', time.gmtime(self.player.get_time() / 1000))
            total_str = time.strftime('%M:%S', time.gmtime(self.player.get_length() / 1000))
            self.time_label.config(text=f"{time_str} / {total_str}")
            
            # Update slider
            if self.player.get_length() > 0:
                self.time_slider.set((self.player.get_time() / self.player.get_length()) * 100)
                
        self.root.after(100, self.update_time)

    def toggle_play(self):
        if hasattr(self, 'player'):
            if self.player.is_playing():
                self.player.pause()
                self.play_button.config(text="▶")
            else:
                self.player.play()
                self.play_button.config(text="⏸")

    def seek_relative(self, seconds):
        if hasattr(self, 'player'):
            current_time = self.player.get_time()
            self.player.set_time(current_time + (seconds * 1000))

    def mark_start(self):
        if hasattr(self, 'player'):
            self.start_time = self.player.get_time()
            self.status_var.set(f"Start marked at {time.strftime('%M:%S', time.gmtime(self.start_time / 1000))}")

    def mark_end(self):
        if hasattr(self, 'player'):
            self.end_time = self.player.get_time()
            self.status_var.set(f"End marked at {time.strftime('%M:%S', time.gmtime(self.end_time / 1000))}")

    def save_clip(self):
        if not hasattr(self, 'player'):
            return
            
        if self.start_time >= self.end_time:
            messagebox.showerror("Error", "Invalid time range")
            return
            
        clip = {
            'url': self.url_entry.get(),
            'start': self.start_time / 1000,
            'end': self.end_time / 1000,
            'transcript': self.transcript_entry.get(),
            'context': self.context_entry.get()
        }
        
        self.clips.append(clip)
        self.save_clips_to_file()
        self.update_clip_list()
        
        # Clear entries
        self.transcript_entry.delete(0, 'end')
        self.context_entry.delete(0, 'end')
        
        self.status_var.set("Clip saved")

    def update_clip_list(self):
        self.clip_tree.delete(*self.clip_tree.get_children())
        for clip in self.clips:
            time_range = f"{time.strftime('%M:%S', time.gmtime(clip['start']))} - {time.strftime('%M:%S', time.gmtime(clip['end']))}"
            self.clip_tree.insert('', 'end', values=(time_range, clip['transcript']))

    def play_selected_clip(self):
        selection = self.clip_tree.selection()
        if not selection:
            return
            
        idx = self.clip_tree.index(selection[0])
        clip = self.clips[idx]
        
        if hasattr(self, 'player'):
            self.player.set_time(int(clip['start'] * 1000))
            self.player.play()

    def delete_selected_clip(self):
        selection = self.clip_tree.selection()
        if not selection:
            return
            
        idx = self.clip_tree.index(selection[0])
        del self.clips[idx]
        self.save_clips_to_file()
        self.update_clip_list()

    def save_clips_to_file(self):
        with open('clips.json', 'w') as f:
            json.dump(self.clips, f, indent=2)

    def load_saved_clips(self):
        try:
            with open('clips.json', 'r') as f:
                self.clips = json.load(f)
        except:
            self.clips = []

    def export_clips(self):
        export_dir = Path('training_data/raw')
        export_dir.mkdir(parents=True, exist_ok=True)
        
        with open('voiceline_timestamps.txt', 'w', encoding='utf-8') as f:
            for i, clip in enumerate(self.clips):
                f.write(f"{clip['url']}|{clip['start']}|{clip['end']}|2b_line_{i:03d}|{clip['transcript']}\n")
        
        self.status_var.set("Clips exported to voiceline_timestamps.txt")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedTimestampTool(root)
    root.mainloop()