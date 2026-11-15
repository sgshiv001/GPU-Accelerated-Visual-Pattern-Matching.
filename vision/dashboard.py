"""Windows dashboard for the visual pattern matching portfolio."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import mkdtemp
import tkinter as tk
from tkinter import messagebox, ttk

VISION = Path(__file__).resolve().parent
PROJECT = VISION.parent
DEMO = VISION / "demo.html"
DLL = PROJECT / "build" / "vision_cuda.dll"

BG = "#10151f"
PANEL = "#192231"
INSET = "#111a27"
RULE = "#324057"
INK = "#eef4fd"
MUTED = "#b7c4d7"
GREEN = "#bded87"
BLUE = "#88d8ef"
PURPLE = "#c1b4ff"
AMBER = "#ffd498"
FONT = "Segoe UI"
MONO = "Consolas"


def label(parent, text="", size=11, color=INK, background=PANEL, bold=False):
    return tk.Label(parent, text=text, fg=color, bg=background, anchor="w",
                    font=(FONT, size, "bold" if bold else "normal"))


def read_snapshot():
    html = DEMO.read_text(encoding="utf-8")
    match = re.search(r'<script id="dataset" type="application/json">(.+?)</script>', html, re.DOTALL)
    if match is None:
        raise ValueError("The saved demo data is missing. Run python vision/make_demo.py to rebuild it.")
    return json.loads(match.group(1))


class Dashboard:
    def __init__(self, root, snapshot):
        self.root = root
        self.samples = snapshot["samples"]
        self.mistakes = [item for item in self.samples if item["svm"] != item["label"]
                         or item["cnn"] != item["label"]]
        self.position = 0
        self.only_mistakes = tk.BooleanVar(root)
        self.recorded = {row["batch"]: row for row in snapshot["timings"]}
        self.live = {}
        self.last_prediction = None
        self.job = None
        self.run_folder = None
        self.operation = None

        root.title("Visual Pattern Matching | CUDA Dashboard")
        root.configure(bg=BG)
        root.minsize(850, 645)
        width = min(1120, max(850, root.winfo_screenwidth() - 100))
        height = min(740, max(645, root.winfo_screenheight() - 100))
        root.geometry(f"{width}x{height}")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.style()

        content = tk.Frame(root, bg=BG)
        content.pack(fill="both", expand=True, padx=23, pady=(17, 10))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(2, weight=1)
        heading = tk.Frame(content, bg=BG)
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        label(heading, "Visual Pattern Matching", 25, background=BG, bold=True).pack(anchor="w")
        label(heading, "Shivanshu Gupta  /  CUDA C++ inference", 11, MUTED, BG).pack(anchor="w", pady=(1, 0))

        stats = tk.Frame(content, bg=BG)
        stats.grid(row=1, column=0, sticky="ew", pady=(0, 13))
        metrics = snapshot["metrics"]
        figures = [
            (f"{metrics['classical']['accuracy']:.2%}", "HOG–PCA–SVM accuracy", PURPLE),
            (f"{metrics['cnn']['accuracy']:.2%}", "CNN accuracy", BLUE),
            (f"{metrics['test_count']} / {metrics['test_count']}", "CUDA predictions matched PyTorch", GREEN),
        ]
        for column, (value, title, color) in enumerate(figures):
            stats.columnconfigure(column, weight=1, uniform="stats")
            tile = self.panel(stats)
            tile.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
            label(tile, value, 21, color, bold=True).pack(anchor="w", padx=13, pady=(5, 0))
            label(tile, title, 10, MUTED).pack(anchor="w", padx=13, pady=(0, 8))

        panels = tk.Frame(content, bg=BG)
        panels.grid(row=2, column=0, sticky="nsew")
        panels.columnconfigure(0, weight=1, minsize=355)
        panels.columnconfigure(1, weight=1, minsize=400)
        panels.rowconfigure(0, weight=1)
        left = self.panel(panels)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        right = self.panel(panels)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.prediction_panel(left)
        self.benchmark_panel(right, snapshot["hardware"]["gpu"])

        footer = tk.Frame(content, bg=BG)
        footer.grid(row=3, column=0, sticky="ew", pady=(11, 0))
        self.status = label(footer, "", 10, MUTED, BG)
        self.status.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=70, takefocus=False)
        self.progress.pack(side="right", padx=(7, 0))
        self.progress.pack_forget()
        ttk.Button(footer, text="Open shareable demo", command=self.open_demo).pack(side="right")
        self.build_button = ttk.Button(footer, text="Build CUDA", command=self.build_cuda)
        self.show_sample()
        self.show_timing()
        self.refresh_actions()
        if DLL.is_file():
            self.set_status("Ready. Select a sample or run a benchmark.")
        else:
            self.set_status("Build CUDA to enable live results. Saved results are available.")

    def style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", background="#253247", foreground=INK, bordercolor=RULE,
                        lightcolor=RULE, darkcolor=RULE, focuscolor=BLUE, font=(FONT, 10),
                        padding=(9, 5))
        style.map("TButton", background=[("active", "#344862"), ("disabled", INSET)],
                  foreground=[("disabled", "#8795ab")])
        style.configure("Accent.TButton", background=GREEN, foreground="#152219",
                        bordercolor=GREEN, font=(FONT, 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#d9f6b7"), ("disabled", "#293a32")],
                  foreground=[("disabled", "#a6b9ab")])
        style.configure("TCheckbutton", background=PANEL, foreground=INK, font=(FONT, 10),
                        indicatorbackground=INSET, focuscolor=BLUE)
        style.map("TCheckbutton", background=[("active", PANEL)],
                  indicatorbackground=[("selected", GREEN), ("!selected", INSET)])
        style.configure("TCombobox", fieldbackground=INSET, background="#253247", foreground=INK,
                        arrowcolor=INK, bordercolor=RULE, font=(FONT, 11), padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", INSET)],
                  foreground=[("readonly", INK)], selectbackground=[("readonly", INSET)],
                  selectforeground=[("readonly", INK)])
        self.root.option_add("*TCombobox*Listbox.background", INSET)
        self.root.option_add("*TCombobox*Listbox.foreground", INK)
        style.configure("TProgressbar", background=GREEN, troughcolor=INSET, bordercolor=RULE)

    def panel(self, parent):
        return tk.Frame(parent, bg=PANEL, highlightbackground=RULE, highlightthickness=1)

    def panel_header(self, panel, title, subtitle):
        label(panel, title, 17, bold=True).pack(anchor="w", padx=13, pady=(9, 0))
        label(panel, subtitle, 10, MUTED).pack(anchor="w", padx=13, pady=(0, 9))

    def prediction_panel(self, panel):
        self.panel_header(panel, "Digit predictions", "Held-out handwritten digits")
        body = tk.Frame(panel, bg=PANEL)
        body.pack(fill="x", padx=13)
        self.digit = tk.Canvas(body, width=134, height=134, bg=INSET, highlightthickness=0,
                               takefocus=False)
        self.digit.pack(side="left", anchor="n", padx=(0, 13))
        answers = tk.Frame(body, bg=PANEL)
        answers.pack(side="left", expand=True, fill="both")
        self.answers = {}
        for key, title in (("label", "True label"), ("svm", "HOG–PCA–SVM"), ("cnn", "CNN")):
            line = tk.Frame(answers, bg=INSET)
            line.pack(fill="both", expand=True, pady=(0, 4))
            label(line, title, 10, MUTED, INSET).pack(side="left", padx=(8, 2))
            value = label(line, "", 15, INK, INSET, bold=True)
            value.pack(side="right", padx=8)
            self.answers[key] = value

        self.sample_text = label(panel, "", 10, MUTED)
        self.sample_text.pack(anchor="w", padx=13, pady=(7, 3))
        navigation = tk.Frame(panel, bg=PANEL)
        navigation.pack(fill="x", padx=13)
        self.previous = ttk.Button(navigation, text="‹ Previous", command=lambda: self.step(-1))
        self.previous.pack(side="left")
        self.next = ttk.Button(navigation, text="Next ›", command=lambda: self.step(1))
        self.next.pack(side="left", padx=5)
        ttk.Checkbutton(navigation, text="Mistakes only", variable=self.only_mistakes,
                        command=self.filter_samples).pack(side="left", padx=(3, 0))

        actions = tk.Frame(panel, bg=PANEL)
        actions.pack(fill="x", padx=13, pady=(10, 4))
        self.predict_button = ttk.Button(actions, text="Predict on CUDA", style="Accent.TButton",
                                         command=self.predict)
        self.predict_button.pack(side="left")
        self.live_label = label(panel, "Live prediction appears here", 10, MUTED)
        self.live_label.pack(anchor="w", padx=13, pady=(0, 8))

    def benchmark_panel(self, panel, device):
        self.panel_header(panel, "Batch performance", "Milliseconds per batch  /  shorter is faster")
        toolbar = tk.Frame(panel, bg=PANEL)
        toolbar.pack(fill="x", padx=13)
        label(toolbar, "Batch", 10, MUTED).pack(side="left", padx=(0, 5))
        values = sorted(self.recorded)
        self.batch = ttk.Combobox(toolbar, values=values, state="readonly", width=6)
        self.batch.set(128 if 128 in values else values[0])
        self.batch.pack(side="left", padx=(0, 7))
        self.batch.bind("<<ComboboxSelected>>", lambda _: self.show_timing())
        self.bench_button = ttk.Button(toolbar, text="Run benchmark", style="Accent.TButton",
                                       command=self.benchmark)
        self.bench_button.pack(side="left")

        self.source_label = label(panel, "", 9, MUTED)
        self.source_label.pack(anchor="w", padx=13, pady=(8, 1))
        chart = tk.Frame(panel, bg=PANEL)
        chart.pack(fill="x", padx=13)
        chart.columnconfigure(1, weight=1)
        self.chart = []
        for row, (title, key, color) in enumerate((
            ("CPU · PyTorch", "cpu_ms", PURPLE),
            ("CUDA · with copies", "fused_wall_ms", GREEN),
            ("CUDA · resident", "fused_gpu_ms", BLUE),
            ("Direct CUDA · resident", "direct_gpu_ms", "#8495b4"),
        )):
            label(chart, title, 9, MUTED).grid(row=row, column=0, sticky="w", padx=(0, 7), pady=3)
            track = tk.Frame(chart, height=9, bg=RULE)
            track.grid(row=row, column=1, sticky="ew", pady=3)
            bar = tk.Frame(track, bg=color)
            bar.place(x=0, y=0, relheight=1)
            value = tk.Label(chart, width=7, anchor="e", bg=PANEL, fg=INK, font=(MONO, 10))
            value.grid(row=row, column=2, sticky="e", padx=(6, 0), pady=3)
            self.chart.append((key, bar, value))
        self.comparison = label(panel, "", 10, INK)
        self.comparison.pack(anchor="w", padx=13, pady=(5, 0))
        short_device = device.replace("GeForce ", "").split(" (SM", 1)[0]
        label(panel, f"Saved results measured on {short_device}", 9, MUTED).pack(
            anchor="w", padx=13, pady=(1, 8))

    def sample_set(self):
        return self.mistakes if self.only_mistakes.get() and self.mistakes else self.samples

    def current_sample(self):
        return self.sample_set()[self.position]

    def step(self, offset):
        self.position += offset
        self.show_sample()

    def filter_samples(self):
        self.position = 0
        self.show_sample()

    def show_sample(self):
        sample = self.current_sample()
        self.digit.delete("all")
        for index, pixel in enumerate(sample["pixels"]):
            y, x = divmod(index, 8)
            color = f"#{24 + 9 * pixel:02x}{32 + 13 * pixel:02x}{42 + 8 * pixel:02x}"
            self.digit.create_rectangle(4 + x * 16, 4 + y * 16, 17 + x * 16, 17 + y * 16,
                                        fill=color, outline="")
        for key, widget in self.answers.items():
            match = sample[key] == sample["label"]
            widget.configure(text=str(sample[key]), fg=INK if key == "label" else GREEN if match else AMBER)
        count = len(self.sample_set())
        self.sample_text.configure(text=f"{self.position + 1} of {count}  ·  Dataset sample #{sample['id']}")
        self.previous.state(["disabled"] if self.position == 0 else ["!disabled"])
        self.next.state(["disabled"] if self.position + 1 == count else ["!disabled"])
        result = self.last_prediction
        if result and result["id"] == sample["id"]:
            self.live_label.configure(text=f"Live CUDA CNN predicted {result['prediction']} · PyTorch match",
                                      fg=GREEN)
        else:
            self.live_label.configure(text="Live prediction appears here", fg=MUTED)

    def show_timing(self):
        size = int(self.batch.get())
        row = self.live.get(size, self.recorded[size])
        maximum = max(row[key] for key, _, _ in self.chart)
        for key, bar, value in self.chart:
            bar.place_configure(relwidth=row[key] / maximum)
            value.configure(text=f"{row[key]:.3f}")
        self.source_label.configure(text="Live measurement" if size in self.live else "Saved measurement",
                                    fg=GREEN if size in self.live else MUTED)
        delta = row["cpu_ms"] / row["fused_wall_ms"]
        faster = f"CUDA {delta:.2f}× faster" if delta >= 1 else f"CPU {1 / delta:.2f}× faster"
        self.comparison.configure(text=f"{faster} with transfers included")

    def set_status(self, text, color=MUTED):
        self.status.configure(text=text, fg=color)

    def refresh_actions(self):
        ready = self.job is None and DLL.is_file()
        for button in (self.predict_button, self.bench_button):
            button.state(["!disabled"] if ready else ["disabled"])
        if not DLL.is_file() or self.operation == "build":
            self.build_button.pack(side="right", padx=(0, 6))
            self.build_button.state(["!disabled"] if self.job is None else ["disabled"])
        else:
            self.build_button.pack_forget()

    def open_demo(self):
        try:
            os.startfile(DEMO)
        except OSError as error:
            messagebox.showerror("Could not open the demo", str(error), parent=self.root)

    def predict(self):
        sample_id = self.current_sample()["id"]
        self.run("prediction", ["--predict-id", str(sample_id), "--prediction-json"],
                 f"Running CUDA prediction for sample #{sample_id}…")

    def benchmark(self):
        size = self.batch.get()
        self.run("benchmark", ["--batches", size], f"Measuring batch {size} on CPU and CUDA…")

    def build_cuda(self):
        self.run("build", [], "Building the native CUDA engine…")

    def run(self, operation, args, status):
        if self.job is not None:
            return
        try:
            run_root = PROJECT / "build" / "dashboard-runs"
            run_root.mkdir(parents=True, exist_ok=True)
            self.run_folder = Path(mkdtemp(prefix=f"{operation}-", dir=run_root))
            python = Path(sys.executable)
            if python.name.lower() == "pythonw.exe":
                python = python.with_name("python.exe")
            command = [str(python), "-u", str(VISION / "benchmark.py"), *args]
            if operation == "benchmark":
                command += ["--output", str(self.run_folder)]
            if operation == "build":
                command = [os.environ.get("ComSpec", "cmd.exe"), "/d", "/c", "build.cmd", "vision"]
            with (self.run_folder / "run.log").open("wb") as log:
                self.job = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as error:
            self.set_status("Could not start the CUDA task.", AMBER)
            messagebox.showerror("Could not start the task", str(error), parent=self.root)
            return
        self.operation = operation
        self.set_status(status, BLUE)
        self.progress.pack(side="right", padx=(7, 0))
        self.progress.start(12)
        self.refresh_actions()
        self.root.after(100, self.poll)

    def poll(self):
        status = self.job.poll()
        if status is None:
            self.root.after(100, self.poll)
            return
        operation, run_folder = self.operation, self.run_folder
        self.job = self.operation = self.run_folder = None
        self.progress.stop()
        self.progress.pack_forget()
        self.refresh_actions()
        try:
            log = (run_folder / "run.log").read_text(encoding="utf-8", errors="replace")
            if status:
                raise RuntimeError(log.strip()[-950:] or f"Process exited with status {status}.")
            if operation == "prediction":
                self.last_prediction = json.loads(log.strip().splitlines()[-1])
                result = self.last_prediction
                self.show_sample()
                self.set_status(f"Sample #{result['id']} verified on CUDA · max logit error "
                                f"{result['max_logit_error']:.2g}", GREEN)
            elif operation == "benchmark":
                with (run_folder / "latency.csv").open(newline="", encoding="utf-8") as file:
                    row = next(csv.DictReader(file))
                size = int(row["batch"])
                self.live[size] = {key: float(value) for key, value in row.items() if key != "batch"}
                self.show_timing()
                self.set_status(f"Batch {size} complete · live timing saved", GREEN)
            elif DLL.is_file():
                self.set_status("CUDA engine built. Live actions are ready.", GREEN)
            else:
                raise RuntimeError("The build exited without creating vision_cuda.dll.")
        except (OSError, ValueError, RuntimeError, IndexError, KeyError, StopIteration) as error:
            self.set_status("The CUDA task failed. See the details for the cause.", AMBER)
            messagebox.showerror("CUDA task failed", f"{error}\n\nLog: {run_folder / 'run.log'}",
                                 parent=self.root)

    def close(self):
        if self.job is not None and self.job.poll() is None:
            self.job.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        Dashboard(root, read_snapshot())
    except (OSError, ValueError, KeyError, IndexError) as error:
        root.withdraw()
        messagebox.showerror("Could not load the dashboard", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
