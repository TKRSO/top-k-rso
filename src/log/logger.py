import pandas as pd
import math
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

class ExperimentLogger:
    def __init__(self, experiment_name, baseline_name="base_iadu"):
        """
        experiment_name: Name of the output file.
        baseline_name: The prefix of the baseline method.
        """
        self.filename = f"{experiment_name}.xlsx"
        self.baseline_name = baseline_name
        self.logs = []
        
        # Metadata columns (Includes "wrf" as requested)
        self.meta_columns = ["shape", "K", "k", "W", "wrf", "g*K/k", "G", "lenCL"]
        
        # Standard metrics suffixes
        self.metric_suffixes = ["_hpfr", "_pss_sum", "_psr_sum", "_rf_sum"]
        self.time_suffixes = ["_prep_time", "_sel_time", "_x_time"]

    def log(self, row_dict):
        """ Logs a result row. """
        self.logs.append(row_dict)

    def _calculate_derived_metrics(self, df):
        """ Calculates Diff% and Speedup relative to the baseline. """
        base_hpfr = f"{self.baseline_name}_hpfr"
        base_time = f"{self.baseline_name}_x_time"
        
        # Identify methods (excluding baseline)
        methods = set()
        for col in df.columns:
            if col.endswith("_hpfr") and col != base_hpfr:
                methods.add(col.replace("_hpfr", ""))
        
        # Calculate Metrics
        for method in methods:
            # --- A. HPFR Diff % ---
            score_col = f"{method}_hpfr"
            diff_col = f"{method}_hpfr_diff%"
            
            if score_col in df.columns and base_hpfr in df.columns:
                df[diff_col] = df.apply(
                    lambda row: (row[score_col] - row[base_hpfr]) / row[base_hpfr] if row[base_hpfr] != 0 else 0,
                    axis=1
                )

            # --- B. Speedup ---
            time_col = f"{method}_x_time"
            speedup_col = f"{method}_speedup"
            
            if time_col in df.columns and base_time in df.columns:
                df[speedup_col] = df.apply(
                    lambda row: row[base_time] / row[time_col] if row[time_col] != 0 else 0,
                    axis=1
                )
                
        return df

    def _organize_columns(self, df):
        """ Organizes columns: Meta -> Baseline -> Methods -> Times """
        cols = list(df.columns)
        final_cols = [c for c in self.meta_columns if c in cols]
        
        # 1. Baseline Metrics
        base_priority = [
            f"{self.baseline_name}_hpfr",
            f"{self.baseline_name}_rf_sum",
            f"{self.baseline_name}_pss_sum",
            f"{self.baseline_name}_psr_sum"
        ]
        base_cols = [c for c in cols if c.startswith(f"{self.baseline_name}_") and not any(t in c for t in ["time", "speedup"])]
        
        for c in base_priority:
            if c in base_cols: final_cols.append(c)
        remaining_base = [c for c in base_cols if c not in final_cols]
        final_cols.extend(sorted(remaining_base))
        
        # 2. Identify Methods
        methods = set()
        for c in cols:
            for suffix in self.metric_suffixes:
                if c.endswith(suffix) and self.baseline_name not in c:
                    methods.add(c.replace(suffix, ""))
        sorted_methods = sorted(list(methods))
        
        # 3. Method Metrics Blocks
        for method in sorted_methods:
            standard_block = [
                f"{method}_hpfr", f"{method}_hpfr_diff%", 
                f"{method}_rf_sum", f"{method}_pss_sum", f"{method}_psr_sum"
            ]
            for col in standard_block:
                if col in cols: final_cols.append(col)
            
            other_cols = [c for c in cols if c.startswith(method) and c not in standard_block and not any(t in c for t in ["time", "speedup"])]
            final_cols.extend(sorted(other_cols))
                    
        # 4. Times & Speedups
        base_times = [c for c in cols if c.startswith(f"{self.baseline_name}_") and "time" in c]
        final_cols.extend(sorted(base_times))
        
        for method in sorted_methods:
            time_block = [
                f"{method}_prep_time", f"{method}_sel_time", 
                f"{method}_x_time", f"{method}_speedup"
            ]
            for col in time_block:
                if col in cols: final_cols.append(col)
                    
        # 5. Catch-all
        existing = set(final_cols)
        final_cols.extend([c for c in cols if c not in existing])
        
        return df[final_cols]

    def save(self):
        if not self.logs:
            print("No logs to save.")
            return

        # 1. Internal Processing (Calculate everything on detailed data first)
        df_detailed = pd.DataFrame(self.logs)
        # We process detailed just to get the columns right before grouping
        df_detailed = self._calculate_derived_metrics(df_detailed)
        
        # 2. Create Summary (Average Sheet)
        # Group by settings (including wrf)
        group_keys = [k for k in ["K", "k", "W", "wrf", "g*K/k", "G"] if k in df_detailed.columns]
        
        if group_keys:
            # Average the data
            df_summary = df_detailed.groupby(group_keys).mean(numeric_only=True).reset_index()
            # Recalculate metrics on the averages
            df_summary = self._calculate_derived_metrics(df_summary)
            df_summary = self._organize_columns(df_summary)
            
            # Remove meaningless columns
            drop_cols = [c for c in ["shape", "lenCL"] if c in df_summary.columns]
            df_summary = df_summary.drop(columns=drop_cols)
        else:
            # Fallback for no groups
            df_summary = pd.DataFrame()

        # 3. Create "Differences Only" Sheet (Subset of Summary)
        if not df_summary.empty:
            # Keep Group Keys
            cols_to_keep = [k for k in group_keys if k in df_summary.columns]
            # Keep Diff% and Speedup columns
            for col in df_summary.columns:
                if "diff%" in col or "speedup" in col:
                    cols_to_keep.append(col)
            
            df_diffs = df_summary[cols_to_keep].copy()
        else:
            df_diffs = pd.DataFrame()

        # 4. Save to Excel
        try:
            with pd.ExcelWriter(self.filename, engine='openpyxl') as writer:
                # SHEET 1: Settings Summary (The Full Averages)
                if not df_summary.empty:
                    df_summary.to_excel(writer, sheet_name='Settings Summary', index=False)
                else:
                    pd.DataFrame(["No data"]).to_excel(writer, sheet_name='Settings Summary')
                
                # SHEET 2: Differences Only (Keys + Diffs/Speedups)
                if not df_diffs.empty:
                    df_diffs.to_excel(writer, sheet_name='Differences Only', index=False)
                
                # NOTE: Detailed Results is NOT saved.
            
            self._apply_pro_styling()
            print(f"✔ Results saved to '{self.filename}'. Sheets: [1] Settings Summary, [2] Differences Only")
            
        except PermissionError:
            print(f"❌ ERROR: Close '{self.filename}' and try again.")
        except Exception as e:
            print(f"❌ ERROR Saving Log: {e}")

    def _apply_pro_styling(self):
        try:
            wb = load_workbook(self.filename)
            
            colors = {
                "header": "404040", "meta": "E7E6E6", "base": "FBE5D6", 
                "score": "E2EFDA", "diff": "D0CECE", "time": "DDEBF7", "speedup": "C6E0B4"
            }
            
            for ws in wb.worksheets:
                # --- NEW LOGIC START: Insert empty row when K OR k changes ---
                K_idx = None
                k_idx = None
                
                # Find column indices for "K" and "k"
                for i, cell in enumerate(ws[1], start=1):
                    if cell.value == "K":
                        K_idx = i
                    elif cell.value == "k":
                        k_idx = i
                
                if K_idx and k_idx:
                    # Start checking from row 3 (comparing to row 2)
                    row = 3
                    while row <= ws.max_row:
                        curr_K = ws.cell(row=row, column=K_idx).value
                        curr_k = ws.cell(row=row, column=k_idx).value
                        
                        prev_K = ws.cell(row=row-1, column=K_idx).value
                        prev_k = ws.cell(row=row-1, column=k_idx).value
                        
                        # Only proceed if we have valid data (prev row isn't an empty separator)
                        if prev_K is not None and curr_K is not None:
                            # If the combination (K, k) changed, insert a row
                            if (curr_K != prev_K) or (curr_k != prev_k):
                                ws.insert_rows(row)
                                row += 2 # Skip the new empty row and the current data row
                            else:
                                row += 1
                        else:
                            row += 1
                # --- NEW LOGIC END ---

                # Headers
                header_fill = PatternFill(start_color=colors["header"], fill_type="solid")
                header_font = Font(bold=True, color="FFFFFF", size=11)
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                
                ws.freeze_panes = "B2"

                # Columns
                for i, col_cells in enumerate(ws.columns, start=1):
                    header_val = str(col_cells[0].value)
                    col_letter = get_column_letter(i)
                    
                    # Width
                    max_len = max([len(str(c.value)) if c.value is not None else 0 for c in col_cells[:15]])
                    ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 40)

                    # Colors
                    fill_color = None
                    if any(x in header_val for x in self.meta_columns): fill_color = colors["meta"]
                    elif self.baseline_name in header_val: fill_color = colors["base"]
                    elif "diff%" in header_val: fill_color = colors["diff"]
                    elif "speedup" in header_val: fill_color = colors["speedup"]
                    elif "time" in header_val: fill_color = colors["time"]
                    elif any(x in header_val for x in ["hpfr", "pss", "psr", "rf_sum"]): fill_color = colors["score"]

                    if fill_color:
                        fill_obj = PatternFill(start_color=fill_color, fill_type="solid")
                        border = Border(left=Side(style='thin', color="BFBFBF"), right=Side(style='thin', color="BFBFBF"))
                        for cell in col_cells[1:]:
                            # Skip styling if it's the empty separator row we just added
                            if cell.value is None: 
                                continue

                            cell.fill = fill_obj
                            cell.border = border
                            
                            if isinstance(cell.value, (int, float)):
                                if "diff%" in header_val: 
                                    cell.number_format = '0.00%'
                                elif "speedup" in header_val: 
                                    cell.number_format = '0.00"x"'
                                elif "time" in header_val:
                                    if cell.value != 0:
                                        val = abs(cell.value)
                                        order = math.floor(math.log10(val))
                                        decimals = max(0, 2 - order)
                                        cell.number_format = '0.' + '0' * decimals
                                    else:
                                        cell.number_format = '0.00'
                                elif "hpfr" in header_val: 
                                    cell.number_format = '0.000'
                                elif abs(cell.value) < 0.01 and cell.value != 0: 
                                    cell.number_format = '0.00E+00'

            wb.save(self.filename)
        except Exception as e:
            print(f"Warning: Styling step failed: {e}")