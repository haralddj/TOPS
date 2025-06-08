import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tops.examples.dyn_models.Stochastic_Constant_Power_Loads.k2aConstPower import ConstPowerLoad


class EigenvaluePlotter:
    def __init__(self, ps_lin):
        self.ps_lin = ps_lin
        self.ps = ps_lin.ps

        self.root = tk.Tk()
        self.root.title("Eigenvalue Plot")

        self.search_frame = tk.Frame(self.root)
        self.search_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.search_entries = []
        self.add_search_field()

        self.add_button = tk.Button(self.search_frame, text="Add Search Field", command=self.add_search_field)
        self.add_button.pack()

        self.update_button = tk.Button(self.search_frame, text="Update Plot", command=self.update_plot)
        self.update_button.pack()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('resize_event', self.on_resize)
        self.plot_eigenvalues()
        self.connect()  # Enable interactivity

    def plot_eigenvalues(self, search_terms=None):
        self.ax.clear()
        self.eigs = self.ps_lin.eigs
        pfs = self.ps_lin.lev.T * self.ps_lin.rev  # Transposed
        self.pfs_abs = np.abs(pfs) / np.max(np.abs(pfs), axis=0)
 

        if search_terms:
            bool_lists = []
            for terms in search_terms:
                bool_list = [any(term in state for term in terms) for state in self.ps.state_desc]
                bool_lists.append(bool_list)

            combined_bool_list = np.logical_and.reduce(bool_lists)
            state_indices = [i for i, flag in enumerate(combined_bool_list) if flag]
            self.opacities = np.max(self.pfs_abs[state_indices, :], axis=0)
        else:
            self.opacities = np.ones(len(self.eigs))

        self.scatter = self.ax.scatter(self.eigs.real, self.eigs.imag, c='b', alpha=self.opacities, s=120, picker=True)
        self.ax.set_xlabel('Real Part')
        self.ax.set_ylabel('Imaginary Part')
        self.ax.set_title('Eigenvalues')
        self.ax.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def connect(self):
        self.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event):
        ind = event.ind[0]  # Get the index of the clicked point
        eig = self.eigs[ind]  # Get the corresponding eigenvalue

        # Calculate frequency (Hz) and damping (%)
        frequency = np.abs(eig.imag) / (2 * np.pi)  # Frequency in Hz
        damping = -eig.real / np.abs(eig) * 100  # Damping in %

        # Add an annotation to the plot
        annotation_text = f"Freq: {frequency:.2f} Hz\nDamping: {damping:.2f} %"
        self.ax.annotate(annotation_text, (eig.real, eig.imag),
                         xytext=(10, 10), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", color='black'),
                         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

        self.fig.canvas.draw()

    def update_plot(self):
        search_terms = []
        for entry in self.search_entries:
            search_term = entry[0].get()
            search_terms.append([term.strip() for term in search_term.split(',')])

        self.plot_eigenvalues(search_terms)

    def add_search_field(self):
        frame = tk.Frame(self.search_frame)
        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT)
        remove_button = tk.Button(frame, text="X", command=lambda: self.remove_search_field(frame))
        remove_button.pack(side=tk.LEFT)
        frame.pack()
        self.search_entries.append((entry, frame))

    def remove_search_field(self, frame):
        for entry, frm in self.search_entries:
            if frm == frame:
                self.search_entries.remove((entry, frm))
                frame.destroy()
                break

    def on_resize(self, event):
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def run(self):
        self.root.mainloop()


def main():
    import tops.examples.user_models.user_lib.k2aTunedToGrid.k2aTuning as model_data
    #import tops.ps_models.k2a as model_data
    #import tops.ps_models.n45_tuned as model_data
    model = model_data.load()


    model['loads'] = {#
         'ConstPowerLoad': [model['loads'][ix] for ix in [0, 1, 2]]
    }

    user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)
    ps.init_dyn_sim()

    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    plotter = EigenvaluePlotter(ps_lin)
    plotter.run()

if __name__ == '__main__':
    main()