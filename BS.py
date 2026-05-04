import re
import os
import h5py
import math
import numpy as np
import pandas as pd
from larch import Group
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ipywidgets import widgets, Layout, link
from scipy.interpolate import PchipInterpolator
from IPython.display import display, clear_output
from typing import Iterable, List, Union, Optional
from larch.xafs import pre_edge, autobk, xftf, cauchy_wavelet

# --- Path Logic ---
#_______________________________________________________________________________
# Part One of the Code
#_______________________________________________________________________________
class ProjectPathNotFound(FileNotFoundError):
    pass

def project_number(project_id: str, session: str = None, subdir: str = None, must_exist: bool = True) -> Path:
    project_id = str(project_id).strip()
    if session:
        path = (Path("/beamlinedata/BIOXAS-MAIN/projects") / f"prj{project_id}" / 
                "raw" / f"session_{session}" / (subdir or ""))
    else:
        path = (Path("/beamlinedata/BIOXAS-SPECTROSCOPY/projects") / f"prj{project_id}" / 
                "main" / "raw" / "acquamanData" / "exportData" / "BioXAS")
    if not path.is_dir():
        path = (Path(r"Z:\projects") / f"prj{project_id}" / 
                "main" / "raw" / "acquamanData" / "exportData" / "BioXAS")
    print("-" * 55)
    print(f"Using path -> {path}")
    print("-" * 55)
    if must_exist and not path.is_dir():
        raise ProjectPathNotFound(f"Directory not found: {path}")
    return path

# --- Deglitcher Code ---

class XASDeglitcher:
    def __init__(self, window=5, threshold=9):
        self.window = int(window)
        self.threshold = float(threshold)

    def _local_stats(self, data):
        n = len(data)
        med = np.zeros(n, dtype=float)
        sigma = np.zeros(n, dtype=float)
        for i in range(self.window, n - self.window):
            local = data[i - self.window:i + self.window + 1]
            median = np.median(local)
            mad = np.median(np.abs(local - median))
            med[i] = median
            sigma[i] = 0.165325 * mad if mad > 0 else 0.0
        return med, sigma

    def _select_region(self, x, region=None, x_range=None, E0=None):
        if region is None: return np.arange(len(x))
        if region == "custom":
            if isinstance(x_range, tuple):
                return np.where((x >= x_range[0]) & (x <= x_range[1]))[0]
            if isinstance(x_range, dict):
                mask = np.zeros(len(x), dtype=bool)
                for r in x_range.values(): mask |= (x >= r[0]) & (x <= r[1])
                return np.where(mask)[0]
        if E0 is None: raise ValueError("E0 required for regions")
        if region == "pre": return np.where(x < E0 - 30)[0]
        if region == "xanes": return np.where((x >= E0 - 30) & (x <= E0 + 40))[0]
        if region == "exafs": return np.where(x > E0 + 100)[0]
        return np.arange(len(x))

    def detect_glitches(self, x, y, region=None, x_range=None, E0=None):
        y, x = np.asarray(y, float), np.asarray(x, float)
        region_idx = set(self._select_region(x, region, x_range, E0))
        glitch_indices = set()
        med, sigma = self._local_stats(y)
        for i in range(self.window, len(y) - self.window):
            if i in region_idx and sigma[i] > 0 and abs(y[i] - med[i]) / sigma[i] > self.threshold:
                glitch_indices.add(i)
        return sorted(list(glitch_indices)), x[sorted(list(glitch_indices))]

    def correct_glitches(self, x, y, glitch_indices):
        mask = np.ones_like(y, dtype=bool)
        mask[glitch_indices] = False
        if mask.sum() < 2: return y.copy()
        # Ensure input is sorted for Pchip
        s_idx = np.argsort(x[mask])
        f = PchipInterpolator(x[mask][s_idx], y[mask][s_idx], extrapolate=False)
        y_corr = y.copy()
        y_corr[~mask] = f(x[~mask])
        valid = np.isfinite(y_corr)
        return np.interp(x, x[valid], y_corr[valid])

    def process(self, x, y, region=None, x_range=None, E0=None, return_indices=False):
        idxs, _ = self.detect_glitches(x, y, region=region, x_range=x_range, E0=E0)
        corrected = self.correct_glitches(x, y, idxs)
        return (corrected, idxs) if return_indices else corrected

class XASLoader:
    def __init__(self, project_id: str, session: str = None):
        self.project_id, self.session = project_id, session
        self._raw_cache, self.multispectra, self.file_selections = {}, {}, {}
        self.selected_files = []
        self.deglitcher = XASDeglitcher() 

        try:
            self.data_path = project_number(self.project_id, self.session)
            self.available_files = sorted([f for f in os.listdir(self.data_path) if f.endswith(('.dat', '.h5'))])
        except:
            self.available_files = []
    def _get_e0(self, x, y):
        """Calculates E0 as the maximum of the first derivative."""
        try:
            dy = np.gradient(y, x)
            e0_idx = np.argmax(dy)
            return x[e0_idx]
        except:
            return None
            
    def gui(self):
        self.out = widgets.Output()
        
        # 1. File Selection
        self.file_sel = widgets.SelectMultiple(
            options=self.available_files, description='Files', 
            layout={'height': '150px', 'width': '600px'}
        )
        self.file_sel.observe(self._on_file_selected, names='value')

        # 2. Detector Identification
        self.elem_text = widgets.Text(value='Zn', description='Element', layout={'width': '20%'})
        self.line_text = widgets.Text(value='Ka1', description='Line', layout={'width': '20%'})
        self.fluo_bank = widgets.Dropdown(options=[('Out', 'OutB'), ('In', 'InB')], value='OutB', description='Board', layout={'width': '25%'})
        detector_box = widgets.HBox([self.elem_text, self.line_text, self.fluo_bank], layout={'width': '100%'})

        # 3. Analysis Mode - Reorganized for visibility
        self.mode_sel = widgets.Dropdown(options=['fluorescence', 'transmission', 'reference'], value='fluorescence', description='Mode', layout={'width': '30%'})
        self.norm_chk = widgets.Checkbox(value=True, description='Normalize', indent=False, layout={'width': '18%'})
        self.sum_chk = widgets.Checkbox(value=True, description='Plot Sum', indent=False, layout={'width': '18%'})
        self.legend_pos = widgets.Dropdown(options=[('Inside', 'best'),('Outside', 'outside')], value='best', description='Legend', layout={'width': '25%'})
        
        # Use HBox with enough width to prevent wrapping
        mode_box = widgets.HBox([self.mode_sel, self.norm_chk, self.sum_chk, self.legend_pos], 
                                layout={'width': '100%', 'justify_content': 'space-between'})

        # 4. Deglitching
        self.deglitch_sel = widgets.Dropdown(
            options=[('None', None), ('Pre-edge', 'pre_edge'), ('EXAFS', 'exafs'), ('XANES', 'xanes'), ('Custom', 'custom')],
            value=None, description='Deglitch', layout={'width': '30%'}
        )
        self.offset_val = widgets.FloatText(value=0.1, description='Offset', layout={'width': '25%'})
        self.custom_range = widgets.Text(value='{"pre": (13000, 13100)}', description='Range', layout={'display': 'none', 'width': '40%'})
        self.deglitch_sel.observe(lambda c: setattr(self.custom_range.layout, 'display', 'block' if c['new'] == 'custom' else 'none'), 'value')
        deglitch_box = widgets.HBox([self.deglitch_sel, self.offset_val, self.custom_range], layout={'width': '100%'})

        # 5. Export
        self.export_style = widgets.Dropdown(options=[('None', 'none'), ('Sum', 'sum'), ('Indiv', 'indiv'), ('All', 'all')], description='💾 Export', layout={'width': '30%'})
        self.export_name = widgets.Text(value='XAS_Data.dat', description='Filename', layout={'width': '65%'})
        export_box = widgets.HBox([self.export_style, self.export_name], layout={'width': '100%'})

        # 6. Run
        self.btn_run = widgets.Button(description="🚀 Run Analysis & Plot", button_style='success', layout={'width': '95%', 'height': '40px'})
        self.btn_run.on_click(self._run_all)


        main_layout = widgets.VBox([
            widgets.HTML("<h2 style='text-align: center; color: #2E86C1;'>BioXAS Data Reduction Dashboard</h2>"),
            widgets.VBox([
                widgets.HTML("<b style='color: #000000'>1. File Selection</b>"), self.file_sel,
                widgets.HTML("<b style='color: #C0392B;'>2. Detector Settings</b>"), detector_box,
                widgets.HTML("<b style='color: #27AE60;'>3. Plotting & Processing</b>"), mode_box,
                widgets.HTML("<b style='color: #2980B9;'>4. Deglitch Options</b>"), deglitch_box,
                widgets.HTML("<b style='color: #8E44AD;'>5. Export Options</b>"), export_box,
                widgets.HTML("<br>"), self.btn_run
            ], layout=widgets.Layout(align_items='center', border='1px solid #ddd', padding='15px', width='700px', margin='auto'))
        ])

        display(main_layout, self.out)

    def _on_file_selected(self, change):
            if not change['new']: 
                self.selected_files = []
                return
            new_selection = list(change['new'])
            old_selection = list(change['old']) if 'old' in change else []
            # Determine what was actually clicked/added
            added = [f for f in new_selection if f not in old_selection]
            # Update the master list
            self.selected_files = new_selection
            if added:
                # If a new file was added via Ctrl+Click, show its grid
                fname = added[-1]
                if fname not in self.file_selections: 
                    self.file_selections[fname] = []
                self._show_inspection_grid(fname)
            else:
                # If a file was removed but others remain, show the grid for the 
                # next available file in the selection
                fname = new_selection[-1]
                self._show_inspection_grid(fname)
                
    def _show_inspection_grid(self, fname):
        self._load_raw_only(fname)
        with self.out:
            clear_output(wait=True)
            if self.mode_sel.value == 'fluorescence':
                plt.close('all')
                fig, axes = plt.subplots(4, 8, figsize=(12, 6), constrained_layout=True)
                ax_map = {axes.flatten()[i]: i+1 for i in range(32)}
                for i in range(32):
                    ax = axes.flatten()[i]
                    ax.plot(self._raw_cache['energy'], (self._raw_cache['val_EC'][:, i]/self._raw_cache['i0']+ 1e-12), color='black', lw=0.5)
                    ax.set_title(f"Ch {i+1}", fontsize=7)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_facecolor('#eaffea' if (i+1) in self.file_selections[fname] else '#ffeaea')
                
                def on_click(event):
                    if event.inaxes in ax_map:
                        ch = ax_map[event.inaxes]
                        if ch in self.file_selections[fname]:
                            self.file_selections[fname].remove(ch)
                            event.inaxes.set_facecolor('#ffeaea')
                        else:
                            self.file_selections[fname].append(ch)
                            event.inaxes.set_facecolor('#eaffea')
                        fig.canvas.draw_idle()
                fig.canvas.mpl_connect('button_press_event', on_click)
                plt.show()
                
    def _run_all(self, b):
        self.multispectra = {}
        with self.out:
            clear_output(wait=True)
            if not self.selected_files: return print("⚠️ No files selected!")
            for fname in self.selected_files:
                self._load_raw_only(fname)
                self._finalize_data(self.mode_sel.value, self.norm_chk.value, fname, self.deglitch_sel.value)
                # Get the data we just processed to find E0
                unique_name = [k for k in self.multispectra.keys() if fname.split('.')[0] in k][-1]
                data = self.multispectra[unique_name]
                e0_val = self._get_e0(data['x'], data['y'])
                if e0_val:
                    print(f"{fname:<40} | {e0_val:.2f}")
                else:
                    print(f"{fname:<40} | Error")
            print("-" * 55)
            self.plot(offset=self.offset_val.value, sum=self.sum_chk.value, legend_loc=self.legend_pos.value)
            if self.export_style.value != 'none': self.export(self.export_name.value, self.export_style.value)

    def _finalize_data(self, mode, norm, fname, dg_mode):
        energy, active = self._raw_cache['energy'], self.file_selections.get(fname, [])
        if mode == 'transmission':
            y = np.log(self._raw_cache['i0'] / (self._raw_cache['i1'] + 1e-12))
            lbl = "trans"
        elif mode == 'reference':
            y = np.log((self._raw_cache['i2'] + 1e-12) /self._raw_cache['i1'])
            lbl = "ref"
        else:
            y = np.nansum(self._raw_cache['val_EC'][:, [c-1 for c in active]], axis=1) / (self._raw_cache['i0']+ 1e-12)
            lbl = f"{self.fluo_bank.value}_{len(active)}ch"
            
        if dg_mode:
            e0 = energy[np.argmax(np.gradient(y, energy))]
            # e0 = e0_val if e0_val is not None else energy[np.argmax(np.gradient(y, energy))]
            y = self.deglitcher.process(energy, y, region=dg_mode, E0=e0)
            
        idx = np.argsort(energy)
        x_f, y_f = energy[idx], y[idx]
        if norm: y_f = (y_f - np.nanmin(y_f)) / (np.nanmax(y_f) - np.nanmin(y_f) + 1e-12)
        self.multispectra[f"{os.path.basename(fname).split('.')[0]}_{lbl}"] = {'x': x_f, 'y': y_f}

    def _load_raw_only(self, fname):
        path = project_number(self.project_id) / fname if fname.endswith('.dat') else project_number(self.project_id, self.session, "epics") / fname
        if fname.endswith('.dat'): self._load_ascii(path)
        else: self._load_h5(path)

    def _load_ascii(self, path):
            self.metadata_header = []
            bank_map = {'InB': {}, 'OutB': {}}
            col_labels = {}
            
            # Metadata storage from old code
            sections = {"top": [], "endstation": [], "roi": [], "scanned": []}
            capture_mode = None
            
            elem, line = self.elem_text.value, self.line_text.value
            
            with open(path, 'r') as f:
                for l_str in f:
                    if l_str.startswith('#'):
                        clean_line = l_str.strip()
                        
                        # --- Metadata Logic (Old Code) ---
                        # 1. Capture XDI and Element tags
                        if any(x in clean_line for x in ["XDI/1.0", "Element.symbol", "Element.edge"]):
                            sections["top"].append(clean_line)
    
                        # 2. Identify Section Triggers
                        if "# Endstation" in clean_line:
                            capture_mode = "endstation"
                        elif "# Regions Of Interest" in clean_line:
                            capture_mode = "roi"
                        elif "# Scanned Regions" in clean_line:
                            capture_mode = "scanned"
                        elif "Column." in clean_line or "Dark Current" in clean_line:
                            capture_mode = None 
    
                        # 3. Categorize lines into sections
                        if capture_mode == "endstation":
                            sections["endstation"].append(clean_line)
                        elif capture_mode == "roi":
                            sections["roi"].append(clean_line)
                        elif capture_mode == "scanned":
                            if any(x in clean_line for x in ["# Scanned Regions", "# Start:", "#"]):
                                sections["scanned"].append(clean_line)
    
                        # --- Column Mapping Logic (New Code) ---
                        if 'Column.' in l_str:
                            parts = l_str.split(':')
                            c_idx, lbl = int(parts[0].split('.')[-1]), parts[-1].strip()
                            col_labels[c_idx] = lbl
                            for b in ['InB', 'OutB']:
                                # Using the regex pattern from your new code
                                m = re.search(fr'{elem}{line}_spectra(\d+)_{b}_DarkCorrect', lbl)
                                if m: 
                                    bank_map[b][int(m.group(1))] = c_idx
                    else: 
                        break
            
            # Combine metadata into the header list
            self.metadata_header = (
                sections["top"] + 
                sections["endstation"] + 
                sections["roi"] + 
                sections["scanned"]
            )
    
            # Data processing (New Code)
            df = pd.read_csv(path, sep=r'\s+', comment='#', header=None).iloc[3:].reset_index(drop=True)
            f0, f1, f2 = self.find_detector_indices([col_labels.get(i, "") for i in range(df.shape[1])])
            
            val_EC = np.zeros((len(df), 32))
            t_bank = bank_map.get(self.fluo_bank.value, {})
            for d_idx, col_idx in t_bank.items():
                if col_idx < df.shape[1]: 
                    val_EC[:, d_idx-1] = df.iloc[:, col_idx].values
            
            self._raw_cache.update({
                'energy': df.iloc[:, 0].values, 
                'i0': df.iloc[:, f0 or 3].values, 
                'i1': df.iloc[:, f1 or 4].values, 
                'i2': df.iloc[:, f2 or 5].values, 
                'val_EC': val_EC
            })

    def _load_h5(self, path):
        with h5py.File(path, 'r') as fe:
            get_p = lambda n: f"/entry/data/{n}"
            det_name = f"{self.elem_text.value}{self.line_text.value}_{self.fluo_bank.value.lower()}"
            try: raw_val = fe[get_p(det_name)][()]
            except: raw_val = fe[get_p("ZnKa1_" + self.fluo_bank.value.lower())][()]
            val_EC = (np.nansum(raw_val, axis=2) if raw_val.ndim == 3 else raw_val)[3:, :]
            self._raw_cache.update({'energy': np.array(fe[get_p('energy')])[3:], 'val_EC': val_EC, 'i0': np.array(fe[get_p('panda_ion_chambers-i0-value')])[3:], 'i1': np.array(fe[get_p('panda_ion_chambers-i1-value')])[3:], 'i2': np.array(fe[get_p('panda_ion_chambers-i2-value')])[3:]})

    def plot(self, **kwargs):
        plt.figure(figsize=(9, 5))
        keys = sorted(self.multispectra.keys())
        if not keys: return
        base_x = self.multispectra[keys[0]]['x']
        all_y = []
        for i, k in enumerate(keys):
            d = self.multispectra[k]
            y_i = np.interp(base_x, d['x'], d['y'])
            all_y.append(y_i)
            plt.plot(d['x'], d['y'] + (i * kwargs.get('offset', 0)), label=k)
        if kwargs.get('sum'):
            plt.plot(base_x, np.mean(all_y, axis=0) + (len(keys) * kwargs.get('offset', 0)), color='black', lw=2, label="SUM")
        if kwargs.get('legend_loc') == 'outside': plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else: plt.legend()
        plt.tight_layout(); plt.show()

    def find_detector_indices(self, column_names):
        m = {'i0': "I0Detector", 'i1': "I1Detector", 'i2': "I2Detector"}
        return [next((i for i, col in enumerate(column_names) if s in col), None) for s in m.values()]

    def export(self, filename, mode):
            if not self.multispectra:
                return print("❌ No data to export.")
                
            keys = sorted(self.multispectra.keys())
            base_x = self.multispectra[keys[0]]['x']
            df_out = pd.DataFrame({'Energy': base_x})
            
            all_y = [np.interp(base_x, self.multispectra[k]['x'], self.multispectra[k]['y']) for k in keys]
            
            if mode in ['indiv', 'all']:
                for i, k in enumerate(keys): 
                    df_out[k] = all_y[i]
            if mode in ['sum', 'all']:
                df_out['Mu'] = np.mean(all_y, axis=0)
    
            # Determine separator: use tabs for .dat, commas for .csv
            sep = '\t' if filename.endswith('.dat') else ','
    
            with open(filename, 'w') as f:
                # Write the filtered metadata header
                if hasattr(self, 'metadata_header'):
                    for line in self.metadata_header:
                        f.write(f"{line}\n")
                
                f.write("#\n") # XDI requirements usually want a # before column headers
                
                # Write data with chosen separator
                df_out.to_csv(f, index=False, sep=sep)
                
                print(f"✅ Exported to {filename} with specific targeted metadata.")

#_______________________________________________________________________________
# Part Two of the Code
#_______________________________________________________________________________
class EXAFSToolkit:
    # def __init__(self, filename):
    #     # Read the file, ignoring lines starting with '#'
    #     # sep=r'\s+' handles both tabs and spaces automatically
    #     df = pd.read_csv(filename, sep=r'\s+', comment='#')
        
    #     self.filename = filename
        
    #     # Pick columns by position instead of names:
    #     # df.iloc[:, 0] is the 1st column (Energy)
    #     # df.iloc[:, 1] is the 2nd column (Mu / Average / Mutrans)
    #     self.group = Group(
    #         energy=df.iloc[:, 0].values, 
    #         mu=df.iloc[:, 4].values, 
    #         groupname='EXAFS'
    #     )
    def __init__(self, filename):
            self.filename = filename
            
            # 1. First, find the header line and index
            header_names = None
            header_idx = 0
            with open(filename, 'r') as f:
                for i, line in enumerate(f):
                    if '# Energy' in line:
                        # Strip the '#' and get clean names like ['Energy', 'I0', ...]
                        header_names = line.lstrip('#').strip().split()
                        header_idx = i
                        break
            
            # 2. Read the file. Use the cleaned names. 
            # Skip everything up to and including the header line to ensure only numbers remain.
            df = pd.read_csv(filename, sep=r'\s+', comment='#', 
                             names=header_names, header=None, 
                             skiprows=header_idx + 1)
            
            # Force numeric conversion to prevent the 'isfinite' TypeError
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
            # 3. Energy is Column 1 (index 0)
            energy_data = df.iloc[:, 0].values
    
            # 4. Check length and search for Mu/Average/Mutrans
            target_keywords = ['mu', 'average', 'mutrans']
            mu_col_index = 1  # Default fallback for 2-column files
            
            if len(df.columns) > 2:
                for i, col_name in enumerate(df.columns):
                    if i == 0: continue # Skip Energy
                    # Check for keyword match in the cleaned headers
                    if any(key in col_name.lower() for key in target_keywords):
                        mu_col_index = i
                        break 
    
            mu_data = df.iloc[:, mu_col_index].values
    
            # 5. Initialize the Group for Larch
            self.group = Group(
                energy=energy_data, 
                mu=mu_data, 
                groupname='EXAFS'
            )

    def process_and_plot(self, tab_idx, mu_type,
                             pre1, pre2, nvict, 
                             norm1, norm2, 
                             rbkg, clamp_lo, clamp_hi,
                             kmin, kmax, kweight, 
                             rmax, window,
                             npre=1, nnorm=1, dk=0.1, rmin=0):
            
            g = self.group
            # 1. Processing
            pre_edge(g, pre1=pre1, pre2=pre2, nvict=nvict, npre=npre, norm1=norm1, norm2=norm2, nnorm=nnorm)
            autobk(g, rbkg=rbkg, kweight=kweight, clamp_lo=clamp_lo, clamp_hi=clamp_hi)
            xftf(g, kmin=kmin, kmax=kmax, kweight=kweight, dk=dk, window=window.lower())
            
            plt.figure(figsize=(10, 5))
            
            # --- TAB 0: Mu(E) (ALL, Normalized, Flat) ---
            if tab_idx == 0:
                if mu_type == 'ALL':
                    plt.plot(g.energy, g.mu, 'k-', label='Raw $\mu(E)$')
                    plt.plot(g.energy, g.pre_edge, 'r--', label='Pre-edge')
                    plt.plot(g.energy, g.post_edge, 'b--', label='Post-edge')
                    plt.plot(g.energy, g.bkg, 'g-', label='Background')
                    # E0 is plotted ONLY here
                    plt.axvline(g.e0, color='orange', linestyle=':', label=f'$E_0$ ({g.e0:.2f} eV)')
                    plt.ylabel("Normalized Intensity (a.u.)") # Renamed from Absorption
                elif mu_type == 'Normalized':
                    plt.plot(g.energy, g.norm, 'b-', label='Normalized')
                    plt.ylabel("Normalized Intensity (a.u.)")
                elif mu_type == 'Flat':
                    plt.plot(g.energy, g.flat, 'g-', label='Flat')
                    plt.ylabel("Normalized Intensity (a.u.)")
                plt.xlabel("Energy (eV)")
                plt.title(f"Mu(E) - {mu_type}")
    
            # --- TAB 1: K-Space (Smoothed) ---
            elif tab_idx == 1:
                k_raw = g.chi * g.k**kweight
                k_smooth = savgol_filter(k_raw, 11, 3)
                plt.plot(g.k, k_smooth, 'b-', lw=1.5, label='Smoothed $\chi(k)$')
                plt.plot(g.k, k_raw, 'k-', alpha=0.2, label='Raw')
                plt.xlabel('$k (\AA^{-1})$')
                plt.ylabel(f'$k^{kweight}\chi(k)$')
    
            # --- TAB 2: R-Space ---
            elif tab_idx == 2:
                plt.plot(g.r, g.chir_mag, 'r-', label='$|\chi(R)|$') # Added label for legend
                plt.xlim(rmin, rmax)
                plt.xlabel('$R (\AA)$')
                plt.ylabel('$|\chi(R)| (\AA^{-%d})$' % (kweight+1)) # Added y-label
                plt.title("Fourier Transform Magnitude")
    
            # --- TAB 3: Wavelet ---
            elif tab_idx == 3:
                cauchy_wavelet(g, kweight=kweight)
                self.plot_wavelet(g)
                return
    
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()
    
    def plot_wavelet(self, g):
        r_grid = getattr(g, "wcauchy_r", getattr(g, "r_w", None))
        k_grid = getattr(g, "k", None)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot Magnitude
        axes[0].imshow(g.wcauchy_mag, extent=[np.min(k_grid), np.max(k_grid), np.min(r_grid), np.max(r_grid)], aspect="auto", origin="lower", cmap='viridis')
        axes[0].set_xlabel('$k (\AA^{-1})$')
        axes[0].set_ylabel('$R (\AA)$')
        axes[0].set_title('Wavelet Magnitude')
        
        # Plot Real Part
        vmax = np.max(np.abs(g.wcauchy_re))
        axes[1].imshow(g.wcauchy_re, extent=[np.min(k_grid), np.max(k_grid), np.min(r_grid), np.max(r_grid)], aspect="auto", origin="lower", cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1].set_xlabel('$k (\AA^{-1})$')
        axes[1].set_ylabel('$R (\AA)$')
        axes[1].set_title('Wavelet Real Part')
        
        plt.tight_layout()
        plt.show()
                
    def _on_export_clicked(self, b):
        """Internal helper to handle button click"""
        g = self.group
        mode = self.export_mode.value
        fname = self.export_name.value
        
        try:
            if mode == 'Normalized Mu':
                df = pd.DataFrame({'Energy': g.energy, 'Normalized_Mu': g.norm, 'Background': g.bkg})
            elif mode == 'K-space (chi)':
                df = pd.DataFrame({'k': g.k, 'chi': g.chi, 'chi_weighted': g.chi * g.k**self.kwt.value})
            elif mode == 'R-space (Mag)':
                df = pd.DataFrame({'R': g.r, 'Chir_Mag': g.chir_mag})
            
            df.to_csv(fname, sep='\t', index=False)
            print(f"Successfully exported to {fname}")
        except Exception as e:
            print(f"Export failed: {e}")

    def gui(self):
        # 1. Standard Tabs
        self.tabs = widgets.Tab(children=[widgets.Output() for _ in range(4)])
        for i, name in enumerate(['Mutran', 'K-space', 'R-space', 'Wavelet']): 
            self.tabs.set_title(i, name)

        # 2. Ghost Controller
        self.tab_selector = widgets.IntSlider(value=0, min=0, max=3)
        link((self.tabs, 'selected_index'), (self.tab_selector, 'value'))
        self.tab_selector.layout.display = 'none'

        # 3. Parameters
        self.mu_type = widgets.Dropdown(options=['ALL', 'Normalized', 'Flat'], value='ALL', description='Mu View', layout={'width': '45%'})
        self.rmax_slider = widgets.FloatSlider(min=2, max=10, value=6.0, description='R-max', layout={'width': '45%'})

        # Row 1: Pre-edge
        self.pre1 = widgets.FloatText(value=-200, description='Pre1', layout={'width': '18%'})
        self.pre2 = widgets.FloatText(value=-30, description='Pre2', layout={'width': '18%'})
        self.nvict = widgets.FloatText(value=1, description='nvict', layout={'width': '20%'})
        self.norm1 = widgets.FloatText(value=150, description='Norm1', layout={'width': '18%'})
        self.norm2 = widgets.FloatText(value=-1, description='Norm2', layout={'width': '18%'})
        pre_box = widgets.HBox([self.pre1, self.pre2, self.nvict, self.norm1, self.norm2])

        # Row 2: Bkg
        self.rbkg = widgets.FloatSlider(min=0.1, max=2.5, step=0.1, value=1.0, description='rbkg', layout={'width': '32%'})
        self.clo = widgets.FloatText(value=2.0, description='Clamp Lo', layout={'width': '32%'})
        self.chi_cl = widgets.FloatText(value=20.0, description='Clamp Hi', layout={'width': '32%'})
        bkg_box = widgets.HBox([self.rbkg, self.clo, self.chi_cl])

        # Row 3: FT
        self.kmin = widgets.FloatText(value=2.0, description='kmin', layout={'width': '23%'})
        self.kmax = widgets.FloatText(value=10.0, description='kmax', layout={'width': '23%'})
        self.kwt = widgets.IntSlider(min=0, max=4, value=2, description='k-weight', layout={'width': '23%'})
        self.win = widgets.Dropdown(options=['Kaiser', 'Hanning'], value='Kaiser', description='Window', layout={'width': '23%'})
        ft_box = widgets.HBox([self.kmin, self.kmax, self.kwt, self.win])

        # 4. EXPORT SECTION
        self.export_mode = widgets.Dropdown(options=['Normalized Mu', 'K-space (chi)', 'R-space (Mag)'], description='Export Type', layout={'width': '40%'})
        self.export_name = widgets.Text(value='Processed_Data.dat', description='Filename', layout={'width': '40%'})
        self.btn_export = widgets.Button(description="💾 Export Data", button_style='info', layout={'width': '15%'})
        self.btn_export.on_click(self._on_export_clicked)
        export_box = widgets.HBox([self.export_mode, self.export_name, self.btn_export], layout={'width': '100%', 'justify_content': 'space-between'})

        # 5. Output logic
        self.out_plot = widgets.interactive_output(self.process_and_plot, {
            'tab_idx': self.tab_selector, 
            'mu_type': self.mu_type,
            'pre1': self.pre1, 
            'pre2': self.pre2, 
            'nvict': self.nvict, 
            'norm1': self.norm1, 
            'norm2': self.norm2,
            'rbkg': self.rbkg, 
            'clamp_lo': self.clo, 
            'clamp_hi': self.chi_cl, 
            'kweight': self.kwt,
            'kmin': self.kmin, 
            'kmax': self.kmax, 
            'window': self.win, 
            'rmax': self.rmax_slider
        })

        # Assemble
        layout = widgets.VBox([
            widgets.HTML("<h2 style='text-align: center; color:  #2E86C1;'>EXAFS Analysis Dashboard</h2>"),
            widgets.VBox([
                self.tabs,
                widgets.HTML("<b style='color: #C0392B;'>1. Pre-edge / Normalization</b>"), pre_box,
                widgets.HTML("<b style='color: #27AE60;'>2. Background Subtraction</b>"), bkg_box,
                widgets.HTML("<b style='color: #2980B9;'>3. Fourier Transform & Display</b>"), ft_box,
                widgets.HBox([self.mu_type, self.rmax_slider], layout={'justify_content': 'space-around'}),
                widgets.HTML("<hr><b style='color: #8E44AD;'>4. Export Processed Data</b>"), export_box,
                widgets.HTML("<br>"),
            ], layout=widgets.Layout(border='1px solid #ddd', padding='15px', width='1000px', margin='auto')),
            self.out_plot
        ])
        display(layout)

