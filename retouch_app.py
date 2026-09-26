#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ferramenta de Retoque Manual e Navegador de Galeria
Desenvolvido para inspeção e retoque de modelos de placas em imagens 3D.
"""

import os
import sys
import ctypes
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

# High-DPI no Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

# ── Tema ─────────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0f0f13",
    "s1":       "#16161c",
    "s2":       "#1c1c24",
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
    "canvas":   "#0a0a0d",
    "sel":      "#7c5cfc",
    "sel_fg":   "#ffffff",
}


class HoverButton(tk.Label):
    """Label que simula botão com hover — visual muito mais limpo que tk.Button."""

    def __init__(self, parent, text="", command=None, bg=None, fg=None,
                 hover_bg=None, font=None, padx=8, pady=3, cursor="hand2", **kw):
        self._bg = bg or C["s3"]
        self._fg = fg or C["text"]
        self._hover_bg = hover_bg or C["hover"]
        self._cmd = command
        super().__init__(
            parent, text=text, bg=self._bg, fg=self._fg,
            font=font or ("Segoe UI", 9), padx=padx, pady=pady,
            cursor=cursor, **kw
        )
        self.bind("<Enter>", lambda e: self.config(bg=self._hover_bg))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda e: self._cmd() if self._cmd else None)

    def set_active(self, active):
        if active:
            self._bg = C["sel"]
            self._fg = C["sel_fg"]
            self._hover_bg = C["accent_h"]
        else:
            self._bg = C["s3"]
            self._fg = C["text"]
            self._hover_bg = C["hover"]
        self.config(bg=self._bg, fg=self._fg)


class RetouchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Retoque de Imagens")
        self.root.geometry("1340x870")
        self.root.minsize(960, 640)
        self.root.configure(bg=C["bg"])

        # Dirs
        self.working_dir = os.path.abspath(
            "Galeria_Limpa" if os.path.exists("Galeria_Limpa") else "test_results"
        )
        self.orig_dir = os.path.abspath("Galeria de imagens")

        # Gallery
        self.image_list = []
        self.current_index = 0
        self.orig_lookup = {}
        self._index_originals()

        # Image state
        self.curr_img_cv = None
        self.is_modified = False
        self.undo_stack = []
        self.redo_stack = []

        # Tool state
        self.active_tool = "eraser"   # eraser | brush | dropper
        self.brush_size = 20
        self.brush_color_hex = "#FFFFFF"
        self.brush_color_bgr = (255, 255, 255)
        self.brush_mask = None
        self.is_drawing = False
        self.last_img_pt = None

        # Zoom/Pan
        self.scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_sx = 0
        self.pan_sy = 0

        # UI refs
        self.tk_image = None
        self._cursor_id = None

        self._build_ui()
        self._bind_keys()
        self.load_working_directory(self.working_dir)

    def _index_originals(self):
        self.orig_lookup = {}
        if os.path.exists(self.orig_dir):
            for r, _, fs in os.walk(self.orig_dir):
                for f in fs:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        self.orig_lookup[f.lower()] = os.path.join(r, f)

    # ═══════════════════════════════════════════════════════════════════
    #   UI BUILD
    # ═══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        root_frame = tk.Frame(self.root, bg=C["bg"])
        root_frame.pack(fill=tk.BOTH, expand=True)

        # ── Header ──────────────────────────────────────────────────────
        hdr = tk.Frame(root_frame, bg=C["s1"], height=42)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        HoverButton(hdr, "📁", self.choose_working_dir, padx=6).pack(
            side=tk.LEFT, padx=(10, 0), pady=5
        )
        self.lbl_folder = tk.Label(
            hdr, text=os.path.basename(self.working_dir),
            bg=C["s1"], fg=C["faint"], font=("Segoe UI", 8)
        )
        self.lbl_folder.pack(side=tk.LEFT, padx=6)

        self._vsep(hdr)

        HoverButton(hdr, "◀", self.prev_image, padx=8).pack(side=tk.LEFT, padx=2, pady=5)
        self.lbl_counter = tk.Label(
            hdr, text="0 / 0", bg=C["s1"], fg=C["cyan"],
            font=("Consolas", 10, "bold"), width=14, anchor="center"
        )
        self.lbl_counter.pack(side=tk.LEFT)
        HoverButton(hdr, "▶", self.next_image, padx=8).pack(side=tk.LEFT, padx=2, pady=5)

        self.lbl_title = tk.Label(
            hdr, text="", bg=C["s1"], fg=C["dim"],
            font=("Segoe UI", 8), anchor="w"
        )
        self.lbl_title.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        self.btn_save = HoverButton(
            hdr, "💾  Salvar", self.save_current_image,
            bg=C["accent"], fg="white", hover_bg=C["accent_h"],
            font=("Segoe UI", 9, "bold"), padx=14
        )
        self.btn_save.pack(side=tk.RIGHT, padx=(2, 10), pady=5)

        HoverButton(
            hdr, "↩  Original", self.revert_to_original,
            bg=C["red_bg"], fg=C["red"], hover_bg="#991b1b",
            font=("Segoe UI", 9), padx=10
        ).pack(side=tk.RIGHT, padx=2, pady=5)

        # Thin rule
        tk.Frame(root_frame, bg=C["border"], height=1).pack(fill=tk.X)

        # ── Toolbar ─────────────────────────────────────────────────────
        tb = tk.Frame(root_frame, bg=C["s2"], height=40)
        tb.pack(fill=tk.X)
        tb.pack_propagate(False)

        # Tool buttons
        self.tbtn_eraser = HoverButton(
            tb, "🧹  Borracha", lambda: self.set_tool("eraser"),
            font=("Segoe UI", 9, "bold"), padx=10
        )
        self.tbtn_eraser.pack(side=tk.LEFT, padx=(10, 2), pady=5)

        self.tbtn_brush = HoverButton(
            tb, "🎨  Pincel", lambda: self.set_tool("brush"),
            font=("Segoe UI", 9, "bold"), padx=10
        )
        self.tbtn_brush.pack(side=tk.LEFT, padx=2, pady=5)

        self.tbtn_dropper = HoverButton(
            tb, "💉", lambda: self.set_tool("dropper"),
            font=("Segoe UI", 10), padx=6
        )
        self.tbtn_dropper.pack(side=tk.LEFT, padx=2, pady=5)

        self._vsep(tb)

        # Brush size
        tk.Label(
            tb, text="Tamanho", bg=C["s2"], fg=C["faint"], font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(4, 2))

        HoverButton(tb, "−", lambda: self.adjust_brush(-4), padx=4, pady=1,
                     font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=1, pady=6)

        self.brush_slider = tk.Scale(
            tb, from_=1, to_=120, orient=tk.HORIZONTAL,
            bg=C["s2"], fg=C["text"], troughcolor=C["border"],
            highlightthickness=0, showvalue=False,
            command=self._on_slider, length=90, width=8, sliderlength=14,
            activebackground=C["accent"]
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=2)

        HoverButton(tb, "+", lambda: self.adjust_brush(4), padx=4, pady=1,
                     font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=1, pady=6)

        self.lbl_size = tk.Label(
            tb, text=f"{self.brush_size}px", bg=C["s2"], fg=C["dim"],
            font=("Consolas", 9), width=5
        )
        self.lbl_size.pack(side=tk.LEFT, padx=4)

        self._vsep(tb)

        # Color section
        tk.Label(
            tb, text="Cor", bg=C["s2"], fg=C["faint"], font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(4, 4))

        # Swatch canvas — shows active color
        self._swatch_frame = tk.Frame(tb, bg=C["border"], padx=1, pady=1)
        self._swatch_frame.pack(side=tk.LEFT, padx=2, pady=8)
        self.swatch = tk.Canvas(
            self._swatch_frame, width=24, height=20, bg=self.brush_color_hex,
            highlightthickness=0, cursor="hand2"
        )
        self.swatch.pack()
        self.swatch.bind("<Button-1>", lambda e: self.choose_color())

        # Palette dots
        palette = [
            "#FFFFFF", "#D4D4D4", "#737373", "#000000",
            "#003399", "#FACC15", "#DC2626",
        ]
        for pc in palette:
            f = tk.Frame(tb, bg=C["border"], padx=1, pady=1)
            f.pack(side=tk.LEFT, padx=1, pady=9)
            dot = tk.Canvas(
                f, width=14, height=14, bg=pc,
                highlightthickness=0, cursor="hand2"
            )
            dot.pack()
            dot.bind("<Button-1>", lambda e, c=pc: self.set_brush_color(c))

        self._vsep(tb)

        # Undo/Redo
        self.btn_undo = HoverButton(tb, "↶  Desfazer", self.undo, padx=6)
        self.btn_undo.pack(side=tk.LEFT, padx=2, pady=5)
        self.btn_redo = HoverButton(tb, "↷  Refazer", self.redo, padx=6)
        self.btn_redo.pack(side=tk.LEFT, padx=2, pady=5)

        self._vsep(tb)

        # Zoom
        HoverButton(tb, "⊞ Fit", self.fit_to_screen, padx=6).pack(
            side=tk.LEFT, padx=2, pady=5
        )
        HoverButton(tb, "1:1", self.zoom_100, padx=6).pack(
            side=tk.LEFT, padx=2, pady=5
        )
        self.lbl_zoom = tk.Label(
            tb, text="100%", bg=C["s2"], fg=C["dim"],
            font=("Consolas", 9), width=5
        )
        self.lbl_zoom.pack(side=tk.LEFT, padx=2)

        # Shortcuts hint (right side)
        tk.Label(
            tb, text="E Borracha  ·  B Pincel  ·  C Cor  ·  Alt+Click Conta-gotas  ·  [ ] Tamanho",
            bg=C["s2"], fg=C["faint"], font=("Segoe UI", 7)
        ).pack(side=tk.RIGHT, padx=10)

        # Rule
        tk.Frame(root_frame, bg=C["border"], height=1).pack(fill=tk.X)

        # ── Canvas ──────────────────────────────────────────────────────
        self.canvas = tk.Canvas(
            root_frame, bg=C["canvas"], highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Alt-ButtonPress-1>", self._eyedropper_click)
        self.canvas.bind("<ButtonPress-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._pan_end)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._pan_end)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self._hide_cursor())

        # ── Status bar ──────────────────────────────────────────────────
        tk.Frame(root_frame, bg=C["border"], height=1).pack(fill=tk.X, side=tk.BOTTOM)
        sb = tk.Frame(root_frame, bg=C["s1"], height=22)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        sb.pack_propagate(False)

        self.lbl_status = tk.Label(
            sb, text="Pronto", bg=C["s1"], fg=C["dim"],
            font=("Segoe UI", 8), anchor="w"
        )
        self.lbl_status.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.lbl_res = tk.Label(
            sb, text="", bg=C["s1"], fg=C["faint"], font=("Consolas", 8)
        )
        self.lbl_res.pack(side=tk.RIGHT, padx=10)

        # Initial tool visual
        self._refresh_tools()

    def _vsep(self, parent):
        tk.Frame(parent, bg=C["border"], width=1).pack(
            side=tk.LEFT, padx=6, pady=6, fill=tk.Y
        )

    def _refresh_tools(self):
        for name, btn in [("eraser", self.tbtn_eraser),
                           ("brush", self.tbtn_brush),
                           ("dropper", self.tbtn_dropper)]:
            btn.set_active(name == self.active_tool)
        self.canvas.config(
            cursor="tcross" if self.active_tool == "dropper" else "crosshair"
        )

    # ═══════════════════════════════════════════════════════════════════
    #   KEYBINDINGS
    # ═══════════════════════════════════════════════════════════════════
    def _bind_keys(self):
        b = self.root.bind
        b("<Left>",         lambda e: self.prev_image())
        b("<Right>",        lambda e: self.next_image())
        b("<a>",            lambda e: self.prev_image())
        b("<d>",            lambda e: self.next_image())
        b("<Control-z>",    lambda e: self.undo())
        b("<Control-y>",    lambda e: self.redo())
        b("<Control-s>",    lambda e: self.save_current_image())
        b("<Control-r>",    lambda e: self.revert_to_original())
        b("<bracketleft>",  lambda e: self.adjust_brush(-4))
        b("<bracketright>", lambda e: self.adjust_brush(4))
        b("<f>",            lambda e: self.fit_to_screen())
        b("<e>",            lambda e: self.set_tool("eraser"))
        b("<E>",            lambda e: self.set_tool("eraser"))
        b("<b>",            lambda e: self.set_tool("brush"))
        b("<B>",            lambda e: self.set_tool("brush"))
        b("<i>",            lambda e: self.set_tool("dropper"))
        b("<I>",            lambda e: self.set_tool("dropper"))
        b("<c>",            lambda e: self.choose_color())
        b("<C>",            lambda e: self.choose_color())

    # ═══════════════════════════════════════════════════════════════════
    #   GALLERY
    # ═══════════════════════════════════════════════════════════════════
    def choose_working_dir(self):
        f = filedialog.askdirectory(initialdir=self.working_dir, title="Pasta de Imagens")
        if f:
            self.load_working_directory(f)

    def load_working_directory(self, folder):
        self.working_dir = os.path.abspath(folder)
        self.lbl_folder.config(text=os.path.basename(self.working_dir))
        exts = (".jpg", ".jpeg", ".png", ".webp")
        self.image_list = []
        for r, _, fs in os.walk(self.working_dir):
            for f in fs:
                if f.lower().endswith(exts):
                    self.image_list.append(os.path.join(r, f))
        self.image_list.sort()
        self.current_index = 0
        if self.image_list:
            self._load_img(0)
        else:
            self.canvas.delete("all")
            self.lbl_counter.config(text="0 / 0")
            self.lbl_title.config(text="Pasta vazia")

    def _load_img(self, idx):
        if not self.image_list or idx < 0 or idx >= len(self.image_list):
            return
        if self.is_modified:
            self.save_current_image(silent=True)
        self.current_index = idx
        fp = self.image_list[idx]
        rel = os.path.relpath(fp, self.working_dir)
        self.lbl_counter.config(text=f"{idx + 1} / {len(self.image_list)}")
        self.lbl_title.config(text=rel)
        try:
            pil = Image.open(fp).convert("RGB")
            self.curr_img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            messagebox.showerror("Erro", str(e))
            return
        h, w = self.curr_img_cv.shape[:2]
        self.lbl_res.config(text=f"{w}×{h}")
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_undo()
        self.is_modified = False
        orig = self._find_orig(fp)
        self.lbl_status.config(
            text=f"Original disponível: {os.path.basename(orig)}" if orig else "Pronto"
        )
        self.fit_to_screen()

    def _find_orig(self, f):
        """Localiza a imagem original pareada na pasta 'Galeria de imagens'."""
        rel = os.path.relpath(f, self.working_dir)
        d = os.path.join(self.orig_dir, rel)
        if os.path.exists(d):
            return d
        fn = os.path.basename(f).lower()
        if fn in self.orig_lookup:
            return self.orig_lookup[fn]
        if fn.startswith("comparacao_"):
            p = fn.replace("comparacao_", "")
            if p in self.orig_lookup:
                return self.orig_lookup[p]
        return None

    def prev_image(self):
        if self.image_list and self.current_index > 0:
            self._load_img(self.current_index - 1)

    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self._load_img(self.current_index + 1)

    # ═══════════════════════════════════════════════════════════════════
    #   TOOLS & COLOR
    # ═══════════════════════════════════════════════════════════════════
    def set_tool(self, name):
        self.active_tool = name
        self._refresh_tools()
        msgs = {
            "eraser":  "🧹 Borracha Mágica ativa",
            "brush":   f"🎨 Pincel ativo · {self.brush_color_hex}",
            "dropper": "💉 Conta-gotas — clique na imagem para capturar cor",
        }
        self.lbl_status.config(text=msgs.get(name, ""))

    def set_brush_color(self, hx):
        if not hx:
            return
        hx = hx.upper()
        c = hx.lstrip("#")
        if len(c) != 6:
            return
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        self.brush_color_hex = hx
        self.brush_color_bgr = (b, g, r)
        self.swatch.config(bg=hx)
        self.set_tool("brush")

    def choose_color(self):
        res = colorchooser.askcolor(color=self.brush_color_hex, title="Cor do Pincel")
        if res and res[1]:
            self.set_brush_color(res[1])

    def _pick_at(self, ix, iy):
        if self.curr_img_cv is None:
            return
        h, w = self.curr_img_cv.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            px = self.curr_img_cv[iy, ix]
            b, g, r = int(px[0]), int(px[1]), int(px[2])
            hx = f"#{r:02X}{g:02X}{b:02X}"
            self.set_brush_color(hx)
            self.lbl_status.config(text=f"✔ Cor capturada: {hx}")

    def _eyedropper_click(self, event):
        if self.curr_img_cv is None:
            return "break"
        ix, iy = self._c2i(event.x, event.y)
        if ix is not None:
            self._pick_at(ix, iy)
        return "break"

    def adjust_brush(self, d):
        v = max(1, min(120, self.brush_size + d))
        self.brush_size = v
        self.brush_slider.set(v)
        self.lbl_size.config(text=f"{v}px")

    def _on_slider(self, val):
        self.brush_size = int(val)
        self.lbl_size.config(text=f"{self.brush_size}px")

    # ═══════════════════════════════════════════════════════════════════
    #   DRAWING
    # ═══════════════════════════════════════════════════════════════════
    def _c2i(self, cx, cy):
        """Canvas coords → image coords."""
        if self.curr_img_cv is None or self.scale <= 0:
            return None, None
        ix = int((cx - self.pan_x) / self.scale)
        iy = int((cy - self.pan_y) / self.scale)
        h, w = self.curr_img_cv.shape[:2]
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))

    def on_left_click(self, event):
        if self.curr_img_cv is None:
            return
        ix, iy = self._c2i(event.x, event.y)
        if ix is None:
            return

        # Dropper: somente se ferramenta ativa é dropper
        # Alt+Click é tratado pelo binding <Alt-ButtonPress-1> separadamente
        if self.active_tool == "dropper":
            self._pick_at(ix, iy)
            self.set_tool("brush")
            return

        self.is_drawing = True
        self.last_img_pt = (ix, iy)
        r = max(1, self.brush_size // 2)

        if self.active_tool == "eraser":
            h, w = self.curr_img_cv.shape[:2]
            self.brush_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(self.brush_mask, (ix, iy), r, 255, -1)
            cr = max(2, int(r * self.scale))
            self.canvas.create_oval(
                event.x - cr, event.y - cr, event.x + cr, event.y + cr,
                fill="#ff0055", outline="", tags="stroke"
            )
        elif self.active_tool == "brush":
            self.push_undo(self.curr_img_cv.copy())
            cv2.circle(self.curr_img_cv, (ix, iy), r, self.brush_color_bgr, -1)
            self.is_modified = True
            self._redraw()
            self._update_cursor(event.x, event.y)

    def on_left_drag(self, event):
        if not self.is_drawing or self.curr_img_cv is None:
            return
        ix, iy = self._c2i(event.x, event.y)
        if ix is None:
            return
        r = max(1, self.brush_size // 2)

        if self.active_tool == "eraser":
            if self.last_img_pt:
                cv2.line(self.brush_mask, self.last_img_pt, (ix, iy), 255, thickness=self.brush_size)
            cv2.circle(self.brush_mask, (ix, iy), r, 255, -1)
            cr = max(2, int(r * self.scale))
            self.canvas.create_oval(
                event.x - cr, event.y - cr, event.x + cr, event.y + cr,
                fill="#ff0055", outline="", tags="stroke"
            )
        elif self.active_tool == "brush":
            if self.last_img_pt:
                cv2.line(self.curr_img_cv, self.last_img_pt, (ix, iy), self.brush_color_bgr, thickness=self.brush_size)
            cv2.circle(self.curr_img_cv, (ix, iy), r, self.brush_color_bgr, -1)
            self._redraw()

        self.last_img_pt = (ix, iy)
        self._update_cursor(event.x, event.y)

    def on_left_release(self, event):
        if not self.is_drawing or self.curr_img_cv is None:
            return
        self.is_drawing = False
        self.last_img_pt = None

        if self.active_tool == "eraser":
            self.canvas.delete("stroke")
            if self.brush_mask is None or np.sum(self.brush_mask > 0) == 0:
                return
            self.push_undo(self.curr_img_cv.copy())
            m = cv2.dilate(self.brush_mask, np.ones((3, 3), np.uint8), iterations=1)
            self.curr_img_cv = cv2.inpaint(self.curr_img_cv, m, 3, cv2.INPAINT_TELEA)
            self.brush_mask = None
            self.is_modified = True
            self.lbl_status.config(text="● Borracha mágica aplicada")
            self._redraw()
        elif self.active_tool == "brush":
            self.is_modified = True
            self.lbl_status.config(text="● Pincel aplicado")

    # ═══════════════════════════════════════════════════════════════════
    #   RESTORE / SAVE
    # ═══════════════════════════════════════════════════════════════════
    def revert_to_original(self):
        """Substitui a imagem atual pela original intacta da Galeria de imagens."""
        if not self.image_list:
            return
        curr = self.image_list[self.current_index]
        orig = self._find_orig(curr)
        if not orig or not os.path.exists(orig):
            messagebox.showwarning("Não encontrada", f"Original não localizada:\n{os.path.basename(curr)}")
            return
        if not messagebox.askyesno("Restaurar", f"Substituir pela original?\n{orig}"):
            return
        self.push_undo(self.curr_img_cv.copy())
        try:
            pil = Image.open(orig).convert("RGB")
            self.curr_img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            self.is_modified = True
            self.lbl_status.config(text="● Restaurada a partir da original")
            self._redraw()
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def save_current_image(self, silent=False):
        """Salva a imagem atualizada no disco com alta qualidade."""
        if self.curr_img_cv is None or not self.image_list:
            return
        t = self.image_list[self.current_index]
        try:
            rgb = cv2.cvtColor(self.curr_img_cv, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(t, quality=95)
            self.is_modified = False
            self.lbl_status.config(text=f"✔ Salva: {os.path.basename(t)}")
            if not silent:
                self.btn_save.config(bg=C["green"], text="✔  Salvo")
                self.btn_save._bg = C["green"]
                self.root.after(1200, lambda: (
                    self.btn_save.config(bg=C["accent"], text="💾  Salvar"),
                    setattr(self.btn_save, '_bg', C["accent"])
                ))
        except Exception as e:
            if not silent:
                messagebox.showerror("Erro", str(e))

    # ═══════════════════════════════════════════════════════════════════
    #   UNDO / REDO
    # ═══════════════════════════════════════════════════════════════════
    def push_undo(self, img):
        self.undo_stack.append(img)
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_undo()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self.curr_img_cv.copy())
        self.curr_img_cv = self.undo_stack.pop()
        self._update_undo()
        self._redraw()
        self.lbl_status.config(text="Desfeito")

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self.curr_img_cv.copy())
        self.curr_img_cv = self.redo_stack.pop()
        self._update_undo()
        self._redraw()
        self.lbl_status.config(text="Refeito")

    def _update_undo(self):
        for btn, stk in [(self.btn_undo, self.undo_stack), (self.btn_redo, self.redo_stack)]:
            if stk:
                btn.config(fg=C["text"])
                btn._fg = C["text"]
            else:
                btn.config(fg=C["faint"])
                btn._fg = C["faint"]

    # ═══════════════════════════════════════════════════════════════════
    #   ZOOM / PAN / RENDER
    # ═══════════════════════════════════════════════════════════════════
    def fit_to_screen(self):
        if self.curr_img_cv is None:
            return
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        ih, iw = self.curr_img_cv.shape[:2]
        self.scale = min((cw - 40) / iw, (ch - 40) / ih)
        self.pan_x = int((cw - iw * self.scale) / 2)
        self.pan_y = int((ch - ih * self.scale) / 2)
        self.lbl_zoom.config(text=f"{int(self.scale * 100)}%")
        self._redraw()

    def zoom_100(self):
        if self.curr_img_cv is None:
            return
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        ih, iw = self.curr_img_cv.shape[:2]
        self.scale = 1.0
        self.pan_x = int((cw - iw) / 2)
        self.pan_y = int((ch - ih) / 2)
        self.lbl_zoom.config(text="100%")
        self._redraw()

    def _on_wheel(self, event):
        if self.curr_img_cv is None:
            return
        f = 1.15 if event.delta > 0 else (1.0 / 1.15)
        ns = max(0.1, min(12.0, self.scale * f))
        mx, my = event.x, event.y
        self.pan_x = int(mx - (mx - self.pan_x) * (ns / self.scale))
        self.pan_y = int(my - (my - self.pan_y) * (ns / self.scale))
        self.scale = ns
        self.lbl_zoom.config(text=f"{int(ns * 100)}%")
        self._redraw()
        self._update_cursor(event.x, event.y)

    def _pan_start(self, event):
        self.is_panning = True
        self.pan_sx = event.x - self.pan_x
        self.pan_sy = event.y - self.pan_y
        self.canvas.config(cursor="fleur")

    def _pan_drag(self, event):
        if not self.is_panning:
            return
        self.pan_x = event.x - self.pan_sx
        self.pan_y = event.y - self.pan_sy
        self._redraw()

    def _pan_end(self, event):
        self.is_panning = False
        self.canvas.config(cursor="tcross" if self.active_tool == "dropper" else "crosshair")

    def _on_resize(self, event):
        if self.curr_img_cv is not None and self.scale == 1.0:
            self.fit_to_screen()
        else:
            self._redraw()

    def _on_motion(self, event):
        self._update_cursor(event.x, event.y)

    def _update_cursor(self, cx, cy):
        if self.active_tool == "dropper":
            self._hide_cursor()
            return
        r = max(2, int((self.brush_size / 2) * self.scale))
        col = self.brush_color_hex if self.active_tool == "brush" else "#38bdf8"
        if self._cursor_id:
            self.canvas.coords(self._cursor_id, cx - r, cy - r, cx + r, cy + r)
            self.canvas.itemconfig(self._cursor_id, outline=col)
            self.canvas.tag_raise(self._cursor_id)
        else:
            self._cursor_id = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=col, width=1, dash=(3, 3), tags="cursor"
            )

    def _hide_cursor(self):
        if self._cursor_id:
            self.canvas.delete(self._cursor_id)
            self._cursor_id = None

    def _redraw(self):
        """Renderiza somente o corte visível da imagem com escala suave e alta velocidade."""
        if self.curr_img_cv is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            return
        ih, iw = self.curr_img_cv.shape[:2]
        x1 = max(0, int(-self.pan_x / self.scale))
        y1 = max(0, int(-self.pan_y / self.scale))
        x2 = min(iw, int((cw - self.pan_x) / self.scale) + 1)
        y2 = min(ih, int((ch - self.pan_y) / self.scale) + 1)
        if x2 <= x1 or y2 <= y1:
            self.canvas.delete("img")
            return
        sl = self.curr_img_cv[y1:y2, x1:x2]
        dw = int((x2 - x1) * self.scale)
        dh = int((y2 - y1) * self.scale)
        if dw <= 0 or dh <= 0:
            return
        rgb = cv2.cvtColor(sl, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        rs = Image.Resampling.NEAREST if self.scale > 3.0 else Image.Resampling.BILINEAR
        pil = pil.resize((dw, dh), resample=rs)
        self.tk_image = ImageTk.PhotoImage(pil)
        dx = int(x1 * self.scale + self.pan_x)
        dy = int(y1 * self.scale + self.pan_y)
        self.canvas.delete("img")
        self.canvas.create_image(dx, dy, anchor=tk.NW, image=self.tk_image, tags="img")
        self.canvas.tag_lower("img")


def main():
    root = tk.Tk()
    RetouchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
