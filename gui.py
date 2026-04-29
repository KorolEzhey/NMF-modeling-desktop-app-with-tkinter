# gui.py
# Desktop-приложение на tkinter для демонстрации NMF

import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data import ARTICLES, TITLES
from nmf_core import difcost, factorize
from preprocessing import build_word_matrix


class NMFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Выделение независимых признаков (NMF)")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        # --- Панель управления (сверху) ---
        ctrl = ttk.Frame(root)
        ctrl.pack(fill="x", padx=10, pady=5)

        self.spin_pc = ttk.Spinbox(ctrl, from_=2, to=10, width=5)
        self.spin_pc.set("4")
        self.spin_pc.pack(side="left", padx=5)

        self.spin_iter = ttk.Spinbox(ctrl, from_=10, to=500, width=6)
        self.spin_iter.set("100")
        self.spin_iter.pack(side="left", padx=5)

        self.btn_run = ttk.Button(ctrl, text="Выделить признаки", command=self.run_nmf)
        self.btn_run.pack(side="left", padx=20)

        self.btn_save = ttk.Button(ctrl, text="Сохранить отчёт", command=self.save_report)
        self.btn_save.pack(side="left")
        self.btn_save.configure(state="disabled")

        # --- Основной контент: слева список текстов, справа вкладки ---
        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        # Левая панель: тексты
        left_frame = ttk.LabelFrame(paned, text="Исходные тексты")
        paned.add(left_frame, weight=1)

        self.listbox = tk.Listbox(left_frame, exportselection=False)
        self.listbox.pack(fill="both", expand=True, padx=5, pady=5)
        for t in TITLES:
            self.listbox.insert("end", t)
        self.listbox.bind("<<ListboxSelect>>", self.on_text_select)

        self.text_preview = scrolledtext.ScrolledText(left_frame, height=6, wrap="word")
        self.text_preview.pack(fill="x", padx=5, pady=5)
        self.text_preview.configure(state="disabled")

        # Правая панель: вкладки
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill="both", expand=True)

        # Вкладка 1: Темы
        self.tab_themes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_themes, text="Темы (признаки)")
        self.txt_themes = scrolledtext.ScrolledText(self.tab_themes, wrap="word")
        self.txt_themes.pack(fill="both", expand=True, padx=5, pady=5)

        # Вкладка 2: Документы
        self.tab_docs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_docs, text="Документы и темы")
        self.txt_docs = scrolledtext.ScrolledText(self.tab_docs, wrap="word")
        self.txt_docs.pack(fill="both", expand=True, padx=5, pady=5)

        # Вкладка 3: График
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="График весов")
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Вкладка 4: Матрица
        self.tab_matrix = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_matrix, text="Матрица")
        self.txt_matrix = scrolledtext.ScrolledText(self.tab_matrix, wrap="word")
        self.txt_matrix.pack(fill="both", expand=True, padx=5, pady=5)

        # Внутренние переменные для сохранения
        self._weights = None
        self._features = None
        self._wordvec = None
        self._titles = None
        self._patternnames = None
        self._toppatterns = None
        self._cost = None

    def on_text_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.text_preview.configure(state="normal")
        self.text_preview.delete("1.0", "end")
        self.text_preview.insert("1.0", ARTICLES[idx])
        self.text_preview.configure(state="disabled")

    def run_nmf(self):
        try:
            pc = int(self.spin_pc.get())
            it = int(self.spin_iter.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры.")
            return

        # 1. Подготовка данных
        matrix, wordvec, titles = build_word_matrix(min_freq=2, max_freq_ratio=0.6)
        if not wordvec:
            messagebox.showerror("Ошибка", "После фильтрации не осталось слов.")
            return

        v = np.matrix(matrix, dtype=float)

        # 2. Запуск NMF
        self.txt_themes.delete("1.0", "end")
        self.root.update_idletasks()

        weights, feat = factorize(v, pc=pc, iter=it)
        wh = weights * feat
        cost = difcost(v, wh)

        self._weights = weights
        self._features = feat
        self._wordvec = wordvec
        self._titles = titles
        self._cost = cost

        # 3. Анализ результатов (showfeatures/showarticles)
        pc_actual, wc = np.shape(feat)
        toppatterns = [[] for _ in range(len(titles))]
        patternnames = []

        themes_out = []
        themes_out.append(f"Итоговая ошибка реконструкции (difcost): {cost:.4f}\n")
        themes_out.append(f"Размерность: документов={len(titles)}, слов={len(wordvec)}, признаков={pc_actual}\n")
        themes_out.append("=" * 60 + "\n")

        for i in range(pc_actual):
            slist = []
            for j in range(wc):
                slist.append((feat[i, j], wordvec[j]))
            slist.sort(reverse=True)
            top_words = [s[1] for s in slist[:8]]
            patternnames.append(top_words)

            themes_out.append(f"Тема {i + 1}: {' | '.join(top_words)}\n")

            flist = []
            for j in range(len(titles)):
                flist.append((weights[j, i], titles[j]))
                toppatterns[j].append((weights[j, i], i, titles[j]))
            flist.sort(reverse=True)

            themes_out.append("  Топ документы:\n")
            for f in flist[:5]:
                themes_out.append(f"    {f[0]:.4f} — {f[1]}\n")
            themes_out.append("\n")

        self._patternnames = patternnames
        self._toppatterns = toppatterns

        self.txt_themes.delete("1.0", "end")
        self.txt_themes.insert("1.0", "".join(themes_out))

        # 4. Документы и их темы (showarticles)
        docs_out = []
        for j in range(len(titles)):
            docs_out.append(f"{titles[j]}\n")
            toppatterns[j].sort(reverse=True)
            for i in range(min(3, pc_actual)):
                score, idx_theme, _ = toppatterns[j][i]
                docs_out.append(f"  {score:.4f} — Тема {idx_theme + 1}: {', '.join(patternnames[idx_theme][:5])}\n")
            docs_out.append("\n")
        self.txt_docs.delete("1.0", "end")
        self.txt_docs.insert("1.0", "".join(docs_out))

        # 5. Матрица (фрагмент)
        mat_out = []
        mat_out.append(f"Матрица V: {np.shape(v)[0]} x {np.shape(v)[1]}\n")
        mat_out.append(f"Матрица W (веса): {np.shape(weights)[0]} x {np.shape(weights)[1]}\n")
        mat_out.append(f"Матрица H (признаки): {np.shape(feat)[0]} x {np.shape(feat)[1]}\n\n")
        mat_out.append("Фрагмент матрицы V (первые 10 слов, первые 5 документов):\n")
        for i in range(min(5, len(titles))):
            row = [int(v[i, j]) for j in range(min(10, len(wordvec)))]
            mat_out.append(f"  {titles[i][:40]:40} {row}\n")
        self.txt_matrix.delete("1.0", "end")
        self.txt_matrix.insert("1.0", "".join(mat_out))

        # 6. График первой темы
        self.plot_theme(0)

        self.btn_save.configure(state="normal")
        messagebox.showinfo("Готово", f"NMF завершён. Ошибка реконструкции: {cost:.4f}")

    def plot_theme(self, theme_idx=0):
        if self._features is None or self._wordvec is None:
            return
        feat = self._features
        wordvec = self._wordvec
        pc_actual, wc = np.shape(feat)
        if theme_idx >= pc_actual:
            theme_idx = 0

        slist = []
        for j in range(wc):
            slist.append((feat[theme_idx, j], wordvec[j]))
        slist.sort(reverse=True)
        top = slist[:10]
        words = [t[1] for t in top]
        vals = [t[0] for t in top]

        self.ax.clear()
        bars = self.ax.barh(range(len(words)), vals, color="steelblue")
        self.ax.set_yticks(range(len(words)))
        self.ax.set_yticklabels(words)
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Вес слова в теме")
        self.ax.set_title(f"Топ-слова темы {theme_idx + 1}")
        margin = max(vals) * 0.15 if vals else 0
        for bar, val in zip(bars, vals):
            self.ax.text(val + margin * 0.3, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=8)
        if vals:
            self.ax.set_xlim(0, max(vals) + margin)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_report(self):
        if self._weights is None:
            return
        try:
            with open("features.txt", "w", encoding="utf-8") as f:
                pc_actual = np.shape(self._features)[0]
                for i in range(pc_actual):
                    f.write(f"Тема {i + 1}: {', '.join(self._patternnames[i])}\n")
                    flist = []
                    for j in range(len(self._titles)):
                        flist.append((self._weights[j, i], self._titles[j]))
                    flist.sort(reverse=True)
                    for score, title in flist[:5]:
                        f.write(f"  {score:.4f} — {title}\n")
                    f.write("\n")
            with open("articles.txt", "w", encoding="utf-8") as f:
                for j in range(len(self._titles)):
                    f.write(f"{self._titles[j]}\n")
                    self._toppatterns[j].sort(reverse=True)
                    for i in range(min(3, pc_actual)):
                        score, idx_theme, _ = self._toppatterns[j][i]
                        f.write(f"  {score:.4f} — Тема {idx_theme + 1}: {', '.join(self._patternnames[idx_theme][:5])}\n")
                    f.write("\n")
            cwd = os.getcwd()
            messagebox.showinfo("Сохранено", f"Отчёты сохранены:\n{cwd}\\features.txt\n{cwd}\\articles.txt")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


def main():
    root = tk.Tk()
    app = NMFApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
