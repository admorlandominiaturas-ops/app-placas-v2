#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Gráfica para Processamento Automatizado de Imagens (Remoção de Placas)
Permite selecionar a pasta de origem, pasta de destino, quantidade de threads/workers,
modo de visualização (preview comparativo ou limpo) e exibe barra de progresso em tempo real.
"""

import os
import sys
import time
import threading
import ctypes
from concurrent.futures import ProcessPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Tentar ativar High-DPI no Windows para nitidez
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# Tema escuro moderno
C = {
    "bg":       "#0f0f13",
    "card":     "#16161c",
    "card2":    "#1c1c24",
    "s3":       "#24242e",
    "border":   "#2c2c38",
    "hover":    "#323240",
    "text":     "#e4e4ec",
    "dim":      "#9898a8",
    "faint":    "#5a5a6e",
    "accent":   "#7c5cfc",
    "accent_h": "#9b7eff",
    "cyan":     "#22d3ee",
    "green":    "#34d399",
    "red":      "#f87171",
    "red_bg":   "#7f1d1d",
    "progress": "#7c5cfc"
}

class ModernButton(tk.Label):
    def __init__(self, parent, text="", command=None, bg=C["accent"], fg="#ffffff", hover_bg=C["accent_h"], font=("Segoe UI", 10, "bold"), pad_x=16, pad_y=8, **kwargs):
        super().__init__(parent, text=text, bg=bg, fg=fg, font=font, cursor="hand2", padx=pad_x, pady=pad_y, **kwargs)
        self.command = command
        self.normal_bg = bg
        self.hover_bg = hover_bg
        self.bind("<Enter>", lambda e: self.config(bg=self.hover_bg))
        self.bind("<Leave>", lambda e: self.config(bg=self.normal_bg))
        self.bind("<Button-1>", lambda e: self.command() if self.command else None)

class ProcessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Processador de Imagens - Ocultador de Placas 3D")
        self.geometry("780x620")
        self.minsize(700, 550)
        self.configure(bg=C["bg"])

        # Variáveis do App
        default_orig = os.path.abspath("Galeria de imagens") if os.path.exists("Galeria de imagens") else ""
        default_dest = os.path.abspath("Galeria_Limpa")
        
        self.orig_dir_var = tk.StringVar(value=default_orig)
        self.dest_dir_var = tk.StringVar(value=default_dest)
        self.workers_var = tk.IntVar(value=os.cpu_count() or 4)
        self.preview_mode_var = tk.BooleanVar(value=False)
        
        self.is_processing = False
        self.stop_requested = False
        self.executor = None

        self._create_layout()

    def _create_layout(self):
        # Header / Título
        header = tk.Frame(self, bg=C["card"], height=70)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        lbl_title = tk.Label(header, text="✨ Processador de Placas 3D", font=("Segoe UI", 15, "bold"), bg=C["card"], fg=C["text"])
        lbl_title.pack(side="left", padx=24, pady=18)

        lbl_sub = tk.Label(header, text="Remoção e ocultação automatizada de caracteres", font=("Segoe UI", 9), bg=C["card"], fg=C["dim"])
        lbl_sub.pack(side="left", pady=22)

        # Body Frame principal
        body = tk.Frame(self, bg=C["bg"], padx=24, pady=20)
        body.pack(fill="both", expand=True)

        # ── 1. Seleção de Pastas ──────────────────────────────────────────────
        sec_pastas = tk.LabelFrame(body, text=" SELEÇÃO DE DIRETÓRIOS ", font=("Segoe UI", 10, "bold"), bg=C["card"], fg=C["accent"], bd=1, relief="solid", padx=16, pady=14)
        sec_pastas.pack(fill="x", pady=(0, 16))

        # Folder Origem
        row1 = tk.Frame(sec_pastas, bg=C["card"])
        row1.pack(fill="x", pady=6)
        tk.Label(row1, text="Pasta Origem (Com Placas):", font=("Segoe UI", 9, "bold"), bg=C["card"], fg=C["text"], width=22, anchor="w").pack(side="left")
        entry_orig = tk.Entry(row1, textvariable=self.orig_dir_var, font=("Segoe UI", 9), bg=C["card2"], fg=C["text"], insertbackground=C["text"], bd=1, relief="solid")
        entry_orig.pack(side="left", fill="x", expand=True, padx=8)
        ModernButton(row1, text="Buscar...", command=self.browse_orig, bg=C["s3"], hover_bg=C["hover"], pad_x=12, pad_y=4).pack(side="right")

        # Folder Destino
        row2 = tk.Frame(sec_pastas, bg=C["card"])
        row2.pack(fill="x", pady=6)
        tk.Label(row2, text="Pasta Destino (Processadas):", font=("Segoe UI", 9, "bold"), bg=C["card"], fg=C["text"], width=22, anchor="w").pack(side="left")
        entry_dest = tk.Entry(row2, textvariable=self.dest_dir_var, font=("Segoe UI", 9), bg=C["card2"], fg=C["text"], insertbackground=C["text"], bd=1, relief="solid")
        entry_dest.pack(side="left", fill="x", expand=True, padx=8)
        ModernButton(row2, text="Buscar...", command=self.browse_dest, bg=C["s3"], hover_bg=C["hover"], pad_x=12, pad_y=4).pack(side="right")

        # ── 2. Configurações ──────────────────────────────────────────────────
        sec_config = tk.LabelFrame(body, text=" CONFIGURAÇÕES DE PROCESSAMENTO ", font=("Segoe UI", 10, "bold"), bg=C["card"], fg=C["accent"], bd=1, relief="solid", padx=16, pady=12)
        sec_config.pack(fill="x", pady=(0, 16))

        row_cfg = tk.Frame(sec_config, bg=C["card"])
        row_cfg.pack(fill="x")

        # Processos / CPU Cores
        tk.Label(row_cfg, text="Núcleos de CPU (Workers):", font=("Segoe UI", 9), bg=C["card"], fg=C["text"]).pack(side="left", padx=(0, 8))
        spin = tk.Spinbox(row_cfg, from_=1, to=os.cpu_count() or 8, textvariable=self.workers_var, width=5, font=("Segoe UI", 9), bg=C["card2"], fg=C["text"], buttonbackground=C["card2"])
        spin.pack(side="left", padx=(0, 32))

        # Checkbox Modo Comparativo
        chk = tk.Checkbutton(row_cfg, text="Salvar com Painel Comparativo (Original / Máscara / Limpa)", variable=self.preview_mode_var, font=("Segoe UI", 9), bg=C["card"], fg=C["text"], selectcolor=C["bg"], activebackground=C["card"], activeforeground=C["text"])
        chk.pack(side="left")

        # ── 3. Status e Log Visual ────────────────────────────────────────────
        sec_log = tk.LabelFrame(body, text=" STATUS E MONITORAMENTO ", font=("Segoe UI", 10, "bold"), bg=C["card"], fg=C["accent"], bd=1, relief="solid", padx=16, pady=12)
        sec_log.pack(fill="both", expand=True, pady=(0, 16))

        # Subframe Progresso
        prog_frame = tk.Frame(sec_log, bg=C["card"])
        prog_frame.pack(fill="x", pady=(0, 8))

        self.lbl_status = tk.Label(prog_frame, text="Aguardando início...", font=("Segoe UI", 9, "bold"), bg=C["card"], fg=C["cyan"])
        self.lbl_status.pack(side="left")

        self.lbl_count = tk.Label(prog_frame, text="0/0 imagens", font=("Segoe UI", 9), bg=C["card"], fg=C["dim"])
        self.lbl_count.pack(side="right")

        # Custom Progressbar / Style
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Custom.Horizontal.TProgressbar", thickness=12, troughcolor=C["card2"], background=C["accent"], bordercolor=C["card"], lightcolor=C["accent"], darkcolor=C["accent"])
        self.progress_bar = ttk.Progressbar(sec_log, style="Custom.Horizontal.TProgressbar", mode="determinate")
        self.progress_bar.pack(fill="x", pady=(0, 10))

        # Caixas de Log
        self.log_text = tk.Text(sec_log, bg=C["card2"], fg=C["text"], font=("Consolas", 8), bd=1, relief="solid", height=6)
        self.log_text.pack(fill="both", expand=True)

        # ── 4. Botões de Ação ──────────────────────────────────────────────────
        btn_frame = tk.Frame(body, bg=C["bg"])
        btn_frame.pack(fill="x", side="bottom")

        self.btn_start = ModernButton(btn_frame, text="🚀 INICIAR PROCESSAMENTO", command=self.start_processing, bg=C["accent"], hover_bg=C["accent_h"], pad_x=24, pad_y=10)
        self.btn_start.pack(side="right")

        self.btn_stop = ModernButton(btn_frame, text="⏹ INTERROMPER", command=self.stop_processing, bg=C["red_bg"], hover_bg=C["red"], pad_x=16, pad_y=10)
        self.btn_stop.pack(side="right", padx=10)
        self.btn_stop.config(state="disabled", bg=C["card2"], fg=C["faint"])

    def browse_orig(self):
        d = filedialog.askdirectory(title="Selecione a Pasta de Origem das Imagens")
        if d:
            self.orig_dir_var.set(os.path.abspath(d))

    def browse_dest(self):
        d = filedialog.askdirectory(title="Selecione a Pasta de Destino das Imagens Processadas")
        if d:
            self.dest_dir_var.set(os.path.abspath(d))

    def log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def start_processing(self):
        orig = self.orig_dir_var.get().strip()
        dest = self.dest_dir_var.get().strip()

        if not orig or not os.path.exists(orig):
            messagebox.showerror("Erro", "Por favor, selecione uma pasta de origem válida.")
            return

        if not dest:
            messagebox.showerror("Erro", "Por favor, especifique a pasta de destino.")
            return

        image_extensions = (".jpg", ".jpeg", ".png", ".webp")
        all_files = []
        for root, _, files in os.walk(orig):
            for f in files:
                if f.lower().endswith(image_extensions):
                    full_in = os.path.join(root, f)
                    rel_path = os.path.relpath(full_in, orig)
                    all_files.append((full_in, rel_path))

        if not all_files:
            messagebox.showwarning("Aviso", f"Nenhuma imagem (.jpg, .png, .webp) foi encontrada em:\n{orig}")
            return

        self.is_processing = True
        self.stop_requested = False
        self.btn_start.config(state="disabled", bg=C["card2"], fg=C["faint"])
        self.btn_stop.config(state="normal", bg=C["red_bg"], fg=C["text"])
        
        self.log_text.delete("1.0", "end")
        self.log(f"[INÍCIO] Mapeadas {len(all_files)} imagens para processamento.")
        self.log(f"[ORIGEM]  {orig}")
        self.log(f"[DESTINO] {dest}\n")

        # Iniciar Worker Thread
        threading.Thread(target=self._run_batch, args=(all_files, orig, dest), daemon=True).start()

    def stop_processing(self):
        if self.is_processing:
            self.stop_requested = True
            self.log("\n[SOLICITAÇÃO] Interrompendo processamento...")

    def _run_batch(self, all_files, orig, dest):
        from run_plate_cleaner import process_single_file

        total = len(all_files)
        workers = max(1, self.workers_var.get())
        preview = self.preview_mode_var.get()

        tasks = []
        for in_path, rel_path in all_files:
            rel_dir = os.path.dirname(rel_path)
            fname = os.path.basename(rel_path)
            out_fname = f"comparacao_{fname}" if preview else fname
            out_path = os.path.join(dest, rel_dir, out_fname)
            tasks.append((in_path, out_path, preview))

        start_time = time.time()
        successes = 0
        plates_found = 0

        self.progress_bar.config(maximum=total, value=0)
        self.lbl_status.config(text="Processando imagens...", fg=C["cyan"])

        with ProcessPoolExecutor(max_workers=workers) as executor:
            self.executor = executor
            futures = [executor.submit(process_single_file, task) for task in tasks]

            for idx, future in enumerate(futures, 1):
                if self.stop_requested:
                    self.log("\n[INTERROMPIDO] Processamento cancelado pelo usuário.")
                    break

                try:
                    ok, fname, val = future.result()
                    if ok:
                        successes += 1
                        plates_found += (1 if val > 0 else 0)
                        status = f"{val} placa(s) ocultada(s)" if val > 0 else "sem placa"
                        msg = f"[{idx:04d}/{total:04d}] {fname}: {status}"
                    else:
                        msg = f"[{idx:04d}/{total:04d}] ERRO em {fname}: {val}"

                    # Atualizar Interface via Event Loop
                    self.after(0, self._update_progress, idx, total, msg)
                except Exception as e:
                    self.after(0, self.log, f"[{idx:04d}/{total:04d}] ERRO: {e}")

        total_time = time.time() - start_time
        self.after(0, self._finish_processing, successes, total, plates_found, total_time)

    def _update_progress(self, current, total, log_msg):
        pct = (current / total) * 100
        self.progress_bar.config(value=current)
        self.lbl_count.config(text=f"{current}/{total} ({pct:.1f}%)")
        self.log(log_msg)

    def _finish_processing(self, successes, total, plates_found, total_time):
        self.is_processing = False
        self.btn_start.config(state="normal", bg=C["accent"], fg="#ffffff")
        self.btn_stop.config(state="disabled", bg=C["card2"], fg=C["faint"])

        if self.stop_requested:
            self.lbl_status.config(text="Processamento Interrompido!", fg=C["red"])
        else:
            self.lbl_status.config(text="Processamento Concluído!", fg=C["green"])
            self.log("\n==========================================")
            self.log(f"FINALIZADO em {total_time:.2f} segundos!")
            self.log(f"Imagens processadas com sucesso: {successes}/{total}")
            self.log(f"Imagens com placas tratadas: {plates_found}")
            self.log("==========================================")
            messagebox.showinfo("Sucesso", f"Processamento concluído com sucesso!\n\nImagens salvas em:\n{self.dest_dir_var.get()}")

if __name__ == "__main__":
    app = ProcessApp()
    app.mainloop()
