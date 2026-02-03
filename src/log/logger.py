import pandas as pd
import re
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

class ExperimentLogger:
    def __init__(self, experiment_name, baseline_name="base_iadu", aggregate_datasets=True):
        self.filename = f"{experiment_name}.xlsx"
        self.baseline_name = baseline_name
        self.aggregate_datasets = aggregate_datasets
        self.logs = []
        
        # Metadata columns
        self.meta_columns = ["K", "k", "W", "wrf", "g*K/k", "G", "shape"]

    def log(self, row_dict):
        """ Logs a single result row. """
        self.logs.append(row_dict)

    def _calculate_derived_metrics(self, df):
        """ Calculates Diff% against Baseline AND Biased vs Stratified. """
        base_hpfr = f"{self.baseline_name}_hpfr"
        base_time = f"{self.baseline_name}_x_time"
        
        # 1. Identify all unique method prefixes (anything ending in _hpfr)
        all_cols = df.columns
        methods = {c.replace("_hpfr", "") for c in all_cols if c.endswith("_hpfr") and self.baseline_name not in c}
        
        for method in methods:
            # --- A. HPFR Diff % (vs Baseline) ---
            score_col = f"{method}_hpfr"
            diff_col = f"{method}_hpfr_diff%"
            
            if score_col in df.columns and base_hpfr in df.columns:
                df[diff_col] = df.apply(
                    lambda row: (row[score_col] - row[base_hpfr]) / row[base_hpfr] if row[base_hpfr] != 0 else 0,
                    axis=1
                )

            # --- B. Speedup (vs Baseline) ---
            time_col = f"{method}_x_time"
            speedup_col = f"{method}_speedup"
            
            if time_col in df.columns and base_time in df.columns:
                df[speedup_col] = df.apply(
                    lambda row: row[base_time] / row[time_col] if row[time_col] != 0 else 0,
                    axis=1
                )

        # 2. Custom: Biased vs Stratified Comparison
        biased_col = "biased_sampling_hpfr"
        strat_col = "stratified_sampling_hpfr"
        custom_diff_col = "biased_vs_stratified_diff%"

        if biased_col in df.columns and strat_col in df.columns:
            df[custom_diff_col] = df.apply(
                lambda row: (row[biased_col] - row[strat_col]) / row[strat_col] if row[strat_col] != 0 else 0,
                axis=1
            )
            
        return df

    def _organize_columns(self, df):
        """ Standardizes column order with strict blocks for every method found. """
        cols = list(df.columns)
        
        # 1. Start with Metadata
        final_cols = [c for c in self.meta_columns if c in cols]
        
        # 2. Baseline Block (HPFR, RF, PSS, PSR) - No Diff%
        base_block = [
            f"{self.baseline_name}_hpfr",
            f"{self.baseline_name}_rf_sum", 
            f"{self.baseline_name}_pss_sum", 
            f"{self.baseline_name}_psr_sum"
        ]
        final_cols.extend([c for c in base_block if c in cols])
        
        # 3. Dynamic Method Blocks
        method_prefixes = sorted({c.replace("_hpfr", "") for c in cols if c.endswith("_hpfr") and self.baseline_name not in c})
        
        for method in method_prefixes:
            block = [
                f"{method}_hpfr", 
                f"{method}_hpfr_diff%", 
                f"{method}_rf_sum", 
                f"{method}_pss_sum", 
                f"{method}_psr_sum"
            ]
            final_cols.extend([c for c in block if c in cols])

        # 4. Custom Comparison Column
        if "biased_vs_stratified_diff%" in cols:
            final_cols.append("biased_vs_stratified_diff%")

        # 5. Times & Speedups (at the end)
        final_cols.extend(sorted([c for c in cols if "time" in c or "speedup" in c]))
        
        # 6. Catch-all
        existing = set(final_cols)
        remaining = [c for c in cols if c not in existing]
        final_cols.extend(remaining)
        
        return df[final_cols]

    def save(self):
        if not self.logs:
            print("No logs to save.")
            return

        df_raw = pd.DataFrame(self.logs)

        # Define Setting keys used for GROUPING
        setting_keys = ["K", "k", "W", "wrf", "g*K/k", "G"]
        
        # If we are NOT aggregating, we must group by shape to keep separate rows
        if not self.aggregate_datasets:
            setting_keys.append("shape")
            
        # Filter keys that actually exist in the data
        setting_keys = [k for k in setting_keys if k in df_raw.columns]
        
        # Aggregation keys for initial sorting
        agg_keys = setting_keys + (["run_id"] if "run_id" in df_raw.columns else [])

        # --- Calculate Metrics BEFORE grouping ---
        df_proc = self._calculate_derived_metrics(df_raw)

        # --- IMPORTANT: Sort by SHAPE first, then by other keys ---
        # This ensures all experiments for Dataset A are grouped together, 
        # allowing "NEXT QUERY" to trigger only when we transition to Dataset B.
        sort_keys = agg_keys.copy()
        if "shape" in sort_keys:
            sort_keys.remove("shape")
            sort_keys.insert(0, "shape")
        
        df_proc = df_proc.sort_values(by=sort_keys)

        # 3. Insert Statistics and Spacers
        final_rows = []
        prev_shape = None

        # --- PART A: Main Results Grouped by Settings ---
        for name, group in df_proc.groupby(setting_keys, sort=False):
            
            # Helper to map the group key(s) back to a dictionary
            if isinstance(name, tuple):
                current_map = dict(zip(setting_keys, name))
            else:
                current_map = {setting_keys[0]: name}
                
            # Helper to restore metadata after mean() operations
            def restore_meta(row_dict):
                for k, v in current_map.items():
                    row_dict[k] = v
                return row_dict

            # --- "NEXT QUERY" Logic ---
            # Trigger ONLY when the 'shape' (Dataset) changes
            current_shape = current_map.get("shape")
            
            if prev_shape is not None and current_shape != prev_shape:
                nq_row = {col: None for col in group.columns}
                nq_row["_row_type"] = "NEXT QUERY"
                final_rows.append(pd.DataFrame([nq_row]))
            
            prev_shape = current_shape

            # 1. Add Data Rows
            if self.aggregate_datasets:
                main_row = group.mean(numeric_only=True).to_dict()
                final_rows.append(pd.DataFrame([restore_meta(main_row)]))
            else:
                # If single run, show it. If multiple, only show stats.
                if len(group) == 1:
                    final_rows.append(group)
            
            # 2. Add Statistics (if meaningful)
            if len(group) > 1:
                target_cols = [c for c in group.columns if "diff%" in c or "speedup" in c]

                mean_row = group.mean(numeric_only=True).to_dict()
                mean_row["_row_type"] = "MEAN"
                final_rows.append(pd.DataFrame([restore_meta(mean_row)]))
                
                max_row = group.max(numeric_only=True)[target_cols].to_dict()
                max_row["_row_type"] = "MAX"
                final_rows.append(pd.DataFrame([restore_meta(max_row)]))
                
                min_row = group.min(numeric_only=True)[target_cols].to_dict()
                min_row["_row_type"] = "MIN"
                final_rows.append(pd.DataFrame([restore_meta(min_row)]))

                std_row = group.std(numeric_only=True)[target_cols].to_dict()
                std_row["_row_type"] = "STD"
                final_rows.append(pd.DataFrame([restore_meta(std_row)]))

            # 3. Add Empty Row Separator (Standard separator between settings)
            sep_row = {col: None for col in group.columns}
            sep_row["_row_type"] = "SEP"
            final_rows.append(pd.DataFrame([sep_row]))

        # --- PART B: Global Statistics by W (or g*K/k) ---
        avg_key = "g*K/k" if "g*K/k" in df_proc.columns else ("W" if "W" in df_proc.columns else None)
        
        if avg_key:
            # Add a big separator
            final_rows.append(pd.DataFrame([{"_row_type": "SEP"}]))
            
            # Group by the primary global key
            w_groups = df_proc.groupby(avg_key)
            
            for w_val, w_group in w_groups:
                
                # Base dictionary for metadata
                base_dict = {m: None for m in self.meta_columns}
                base_dict[avg_key] = w_val
                # Try to preserve W if we are grouping by g*K/k
                if "W" in self.meta_columns and avg_key != "W":
                     base_dict["W"] = w_group["W"].iloc[0] if "W" in w_group.columns else None

                # 1. W-AVG
                avg_row = w_group.mean(numeric_only=True).to_dict()
                avg_row.update(base_dict)
                avg_row["_row_type"] = "W-AVG"
                final_rows.append(pd.DataFrame([avg_row]))

                # 2. W-MAX
                max_row = w_group.max(numeric_only=True).to_dict()
                max_row.update(base_dict)
                max_row["_row_type"] = "W-MAX"
                final_rows.append(pd.DataFrame([max_row]))

                # 3. W-MIN
                min_row = w_group.min(numeric_only=True).to_dict()
                min_row.update(base_dict)
                min_row["_row_type"] = "W-MIN"
                final_rows.append(pd.DataFrame([min_row]))

                # 4. W-STD
                std_row = w_group.std(numeric_only=True).to_dict()
                std_row.update(base_dict)
                std_row["_row_type"] = "W-STD"
                final_rows.append(pd.DataFrame([std_row]))

                # Spacer between Global Groups
                final_rows.append(pd.DataFrame([{"_row_type": "SEP"}]))

        df_final = pd.concat(final_rows, ignore_index=True)
        df_final = self._organize_columns(df_final)

        # 4. Handle Type Column
        if "_row_type" in df_final.columns:
            df_final.insert(0, "Type", df_final["_row_type"].fillna(""))
            df_final.drop(columns=["_row_type"], inplace=True)
        else:
            df_final.insert(0, "Type", "")

        # 5. Cleanup - FORCE REMOVAL OF SHAPE
        cols_to_drop = ["run_id", "lenCL", "shape"] 
        df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns], inplace=True)

        # 6. Create Second DataFrame for Diff Only
        diff_cols = [c for c in df_final.columns if "diff%" in c]
        meta_cols_present = [c for c in self.meta_columns if c in df_final.columns]
        
        diff_sheet_cols = ["Type"] + meta_cols_present + diff_cols
        df_diff = df_final[diff_sheet_cols].copy()

        # 7. Save to Excel
        try:
            with pd.ExcelWriter(self.filename, engine='openpyxl') as writer:
                if not df_final.empty:
                    df_final_clean = df_final.copy()
                    df_final_clean["Type"] = df_final_clean["Type"].replace("SEP", "")
                    df_final_clean.to_excel(writer, sheet_name='Summary', index=False)
                
                if not df_diff.empty:
                    df_diff_clean = df_diff.copy()
                    df_diff_clean["Type"] = df_diff_clean["Type"].replace("SEP", "")
                    df_diff_clean.to_excel(writer, sheet_name='Diffs', index=False)
            
            self._apply_pro_styling()
            print(f"✔ Results saved to '{self.filename}'.")
        
        except PermissionError:
            print(f"❌ ERROR: Close '{self.filename}' and try again.")
        except Exception as e:
            print(f"❌ ERROR Saving Log: {e}")

    def _apply_pro_styling(self):
        try:
            wb = load_workbook(self.filename)
            
            # --- COLOR PALETTE ---
            c_header = "404040"
            c_meta   = "E7E6E6"

            # 3. Method Rotation Palette (Soft Pastel)
            method_palette = [
                "E2EFDA", # Light Green
                "FBE5D6", # Light Orange
                "DDEBF7", # Light Blue
                "FFF2CC", # Light Yellow
                "E2F0D9", # Mint
                "D9D9D9", # Light Grey
            ]

            # Helper to Map Column Names -> Method Names
            suffixes = ["_hpfr", "_rf_sum", "_pss_sum", "_psr_sum", "_x_time", "_speedup", "_diff%"]
            
            def get_method_name(header_val):
                if not header_val: return "meta"
                h = str(header_val)
                if h == "Type" or h in self.meta_columns: return "meta"
                if "biased_vs_stratified" in h: return "comparison"
                
                root = h
                for s in suffixes:
                    if root.endswith(s):
                        root = root.replace(s, "")
                        break
                return root

            for ws in wb.worksheets:
                col_map = {cell.value: i for i, cell in enumerate(ws[1], 1)}
                type_idx = col_map.get("Type")

                ws.freeze_panes = "C2" if type_idx else "B2"
                
                # 1. Identify Unique Methods
                headers = [cell.value for cell in ws[1]]
                unique_methods = []
                for h in headers:
                    m = get_method_name(h)
                    if m not in unique_methods and m != "meta":
                        unique_methods.append(m)
                
                # Create Color Map
                method_color_map = {}
                for idx, m_name in enumerate(unique_methods):
                    color = method_palette[idx % len(method_palette)]
                    method_color_map[m_name] = color
                
                method_color_map["meta"] = c_meta

                # 2. Styling Loop
                for i, col_cells in enumerate(ws.columns, 1):
                    header = str(col_cells[0].value)
                    col_letter = get_column_letter(i)
                    
                    # Header
                    head_cell = col_cells[0]
                    head_cell.fill = PatternFill(start_color=c_header, fill_type="solid")
                    head_cell.font = Font(bold=True, color="FFFFFF")
                    head_cell.alignment = Alignment(horizontal="center", vertical="center")

                    # Auto-width
                    valid_vals = [len(str(c.value)) for c in col_cells[:20] if c.value is not None]
                    max_len = max(valid_vals) if valid_vals else 0
                    ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 35)

                    # Determine Column Background Color
                    m_name = get_method_name(header)
                    col_bg_hex = method_color_map.get(m_name, "FFFFFF") 
                    col_fill = PatternFill(start_color=col_bg_hex, fill_type="solid")

                    thin_border = Border(left=Side(style='thin', color="BFBFBF"), right=Side(style='thin', color="BFBFBF"))

                    for r_idx, cell in enumerate(col_cells[1:], 2):
                        is_empty_row = all(ws.cell(r_idx, c).value in [None, ""] for c in range(1, ws.max_column + 1))
                        if is_empty_row: 
                            continue

                        r_type = ws.cell(r_idx, type_idx).value if type_idx else None
                        
                        # Skip blank cells unless it's a styled row
                        if cell.value is None and r_type not in ["MEAN", "MAX", "MIN", "STD", "W-AVG", "W-MAX", "W-MIN", "W-STD", "NEXT QUERY"]: 
                            continue 
                        
                        cell.border = thin_border
                        
                        # --- CRITICAL FIX: ALWAYS apply Column Color to Background ---
                        cell.fill = col_fill 

                        # --- Apply Font Styles for Stats (No background overrides) ---
                        if r_type == "MEAN": cell.font = Font(bold=True, color="1F4E78")
                        elif r_type == "MAX": cell.font = Font(italic=True, color="833C0C")
                        elif r_type == "MIN": cell.font = Font(italic=True, color="833C0C")
                        elif r_type == "STD": cell.font = Font(italic=True, color="9C5700")
                        elif r_type == "NEXT QUERY":
                            cell.font = Font(bold=True, color="FF0000")
                            cell.alignment = Alignment(horizontal="left")
                        
                        # GLOBAL STATS STYLES
                        elif r_type == "W-AVG": cell.font = Font(bold=True, color="203764")
                        elif r_type == "W-MAX": cell.font = Font(italic=True, color="542914")
                        elif r_type == "W-MIN": cell.font = Font(italic=True, color="542914")
                        elif r_type == "W-STD": cell.font = Font(italic=True, color="542914")

                        # Number Formats
                        if isinstance(cell.value, (int, float)):
                            if "diff%" in header: cell.number_format = '0.00%'
                            elif "speedup" in header: cell.number_format = '0.00"x"'
                            elif "time" in header: cell.number_format = '0' if abs(cell.value) > 1000 else '0.00'
                            elif "hpfr" in header: cell.number_format = '0.000'
                            elif abs(cell.value) < 0.01 and cell.value != 0: cell.number_format = '0.00E+00'

            wb.save(self.filename)
        except Exception as e:
            print(f"Warning: Styling step failed: {e}")