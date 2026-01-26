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
        
        # Metadata columns (Removed "shape" and "run_id" from final output priority)
        self.meta_columns = ["K", "k", "W", "wrf", "g*K/k", "G"]
        
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

        # 1. Calculate metrics on detailed logs
        df_detailed = pd.DataFrame(self.logs)
        df_detailed = self._calculate_derived_metrics(df_detailed)
        
        # 2. AGGREGATION: Average distinct datasets (shapes) for the SAME Run ID
        # We group by the full setting signature INCLUDING run_id
        agg_keys = ["K", "k", "W", "wrf", "g*K/k", "G", "run_id"]
        agg_keys = [k for k in agg_keys if k in df_detailed.columns]
        
        if agg_keys:
            df_aggregated = df_detailed.groupby(agg_keys, as_index=False).mean(numeric_only=True)
        else:
            df_aggregated = df_detailed

        # 3. STD DEV ROW: Insert a standard deviation row after each SET of Runs
        # Group keys represent the Experiment Setting (excluding run_id)
        setting_keys = [k for k in agg_keys if k != "run_id"]
        
        df_final = pd.DataFrame()

        if setting_keys:
            final_rows = []
            # Sort to keep runs ordered (1, 2, 3...) within settings
            df_aggregated = df_aggregated.sort_values(by=agg_keys)
            
            # Iterate over each setting group (contains Run 1, Run 2...)
            for name, group in df_aggregated.groupby(setting_keys, sort=False):
                # Add the individual Run rows (which are averages of datasets)
                final_rows.append(group)
                
                # Calculate Std Dev Row if we have multiple runs
                if len(group) > 1:
                    # Get standard deviation for numeric columns
                    std_series = group.std(numeric_only=True)
                    
                    # --- FILTER: Keep only 'diff%' columns ---
                    target_cols = [c for c in std_series.index if "diff%" in c]
                    std_row = std_series[target_cols].to_dict()
                    
                    # Restore Metadata for alignment
                    if len(setting_keys) == 1:
                        std_row[setting_keys[0]] = name
                    else:
                        for k, v in zip(setting_keys, name):
                            std_row[k] = v
                    
                    # Mark this row explicitly (using a temp column we'll drop later or styling key)
                    # We use a special marker in the first string column if possible, or just append
                    # Since we are dropping run_id later, we can't use it for the final display text easily
                    # unless we keep a marker column.
                    std_row["_row_type"] = "STD_DEV" 
                    
                    # Append std dev row
                    final_rows.append(pd.DataFrame([std_row]))
            
            # Combine everything
            df_final = pd.concat(final_rows, ignore_index=True)
        else:
            df_final = df_aggregated

        # 4. Cleanup
        df_final = self._organize_columns(df_final)
        
        # Explicitly remove unwanted columns
        drop_cols = [c for c in ["shape", "lenCL", "run_id"] if c in df_final.columns]
        df_final = df_final.drop(columns=drop_cols)

        # 5. Create "Differences Only" Sheet (Subset)
        cols_to_keep = [k for k in setting_keys if k in df_final.columns]
        # We need to keep the row type marker if it exists to style it, but we don't want to print it
        # Actually, let's keep it for now to help styling, then hide it? 
        # Better strategy: Use the fact that numeric columns are empty in the STD row to identify it, 
        # or check the specific 'diff%' values. 
        
        for col in df_final.columns:
            if "diff%" in col or "speedup" in col:
                cols_to_keep.append(col)
        
        # Ensure _row_type is kept if it exists for splitting logic
        if "_row_type" in df_final.columns:
            cols_to_keep.append("_row_type")

        df_diffs = df_final[cols_to_keep].copy() if not df_final.empty else pd.DataFrame()

        # 6. Save to Excel
        try:
            # We remove _row_type before saving to excel, but we need to track indices for styling
            # Alternatively, we leave it and hide it. Let's remove it and use a trick:
            # The row where 'K' is empty (nan) but has data is the STD row? 
            # No, we filled K. 
            
            # Let's add a "Stat" column to the start if we want to be explicit? 
            # Or just color it.
            
            # To enable styling, we will rely on the fact that only "diff%" columns are populated 
            # in the STD rows, while other numeric columns (like hpfr) are NaN.
            
            # Drop the internal marker before writing, but we need to know which rows are which.
            # Let's add a temporary column "Row Type" to the output, style it, then maybe the user won't mind?
            # User said "remove row id column is useless". 
            
            # I will repurpose the first column to say "STD" if it's a variance row?
            # No, that messes up filtering.
            
            # Best approach: Leave the DataFrame as is, but in styling, detect the STD row via `_row_type`.
            # However, `to_excel` writes the dataframe.
            
            # Let's add a visual marker in a new column "Metric" that is usually blank but "STD" for these rows.
            cols = list(df_final.columns)
            if "K" in cols:
                loc_idx = cols.index("K")
                df_final.insert(loc_idx, "Type", "")
                df_diffs.insert(0, "Type", "")
            else:
                df_final["Type"] = ""
                df_diffs["Type"] = ""

            if "_row_type" in df_final.columns:
                df_final.loc[df_final["_row_type"] == "STD_DEV", "Type"] = "STD"
                df_final = df_final.drop(columns=["_row_type"])
                
            if "_row_type" in df_diffs.columns:
                df_diffs.loc[df_diffs["_row_type"] == "STD_DEV", "Type"] = "STD"
                df_diffs = df_diffs.drop(columns=["_row_type"])

            with pd.ExcelWriter(self.filename, engine='openpyxl') as writer:
                if not df_final.empty:
                    df_final.to_excel(writer, sheet_name='Settings Summary', index=False)
                else:
                    pd.DataFrame(["No data"]).to_excel(writer, sheet_name='Settings Summary')
                
                if not df_diffs.empty:
                    df_diffs.to_excel(writer, sheet_name='Differences Only', index=False)
            
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
                "score": "E2EFDA", "diff": "D0CECE", "time": "DDEBF7", "speedup": "C6E0B4",
                "std_dev": "FFEB9C" # Light Yellow/Gold for Standard Deviation rows
            }
            
            for ws in wb.worksheets:
                # Identify columns
                K_idx = None
                k_idx = None
                type_idx = None
                
                for i, cell in enumerate(ws[1], start=1):
                    val = str(cell.value)
                    if val == "K": K_idx = i
                    elif val == "k": k_idx = i
                    elif val == "Type": type_idx = i
                
                # Insert Separator Rows (Logic: If K or k changes)
                if K_idx and k_idx:
                    row = 3
                    while row <= ws.max_row:
                        curr_K = ws.cell(row=row, column=K_idx).value
                        curr_k = ws.cell(row=row, column=k_idx).value
                        prev_K = ws.cell(row=row-1, column=K_idx).value
                        prev_k = ws.cell(row=row-1, column=k_idx).value
                        
                        # Only separate if valid data and not a STD row (STD rows share the same K/k)
                        if prev_K is not None and curr_K is not None:
                            if (curr_K != prev_K) or (curr_k != prev_k):
                                ws.insert_rows(row)
                                row += 2 
                            else:
                                row += 1
                        else:
                            row += 1

                # Styling
                header_fill = PatternFill(start_color=colors["header"], fill_type="solid")
                header_font = Font(bold=True, color="FFFFFF", size=11)
                for cell in ws[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                
                ws.freeze_panes = "B2"

                for i, col_cells in enumerate(ws.columns, start=1):
                    header_val = str(col_cells[0].value)
                    col_letter = get_column_letter(i)
                    
                    # Width
                    max_len = max([len(str(c.value)) if c.value is not None else 0 for c in col_cells[:15]])
                    ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 40)

                    # Determine Base Color for Column
                    base_fill_color = None
                    if any(x in header_val for x in self.meta_columns) or header_val == "Type": base_fill_color = colors["meta"]
                    elif self.baseline_name in header_val: base_fill_color = colors["base"]
                    elif "diff%" in header_val: base_fill_color = colors["diff"]
                    elif "speedup" in header_val: base_fill_color = colors["speedup"]
                    elif "time" in header_val: base_fill_color = colors["time"]
                    elif any(x in header_val for x in ["hpfr", "pss", "psr", "rf_sum"]): base_fill_color = colors["score"]

                    if base_fill_color:
                        fill_obj = PatternFill(start_color=base_fill_color, fill_type="solid")
                        std_fill = PatternFill(start_color=colors["std_dev"], fill_type="solid")
                        border = Border(left=Side(style='thin', color="BFBFBF"), right=Side(style='thin', color="BFBFBF"))
                        
                        for r_idx, cell in enumerate(col_cells[1:], start=2):
                            if cell.value is None: continue

                            # Check if this is a STD row using the "Type" column
                            is_std = False
                            if type_idx:
                                val = ws.cell(row=r_idx, column=type_idx).value
                                if val == "STD":
                                    is_std = True
                            
                            cell.border = border
                            
                            # Apply color: STD color takes precedence
                            if is_std:
                                cell.fill = std_fill
                                cell.font = Font(italic=True, bold=True, color="9C5700") # Darker gold text
                            else:
                                cell.fill = fill_obj
                            
                            # Number Formats
                            if isinstance(cell.value, (int, float)):
                                if "diff%" in header_val: 
                                    cell.number_format = '0.00%'
                                elif "speedup" in header_val: 
                                    cell.number_format = '0.00"x"'
                                elif "time" in header_val:
                                    if cell.value != 0 and abs(cell.value) < 1000:
                                        val = abs(cell.value)
                                        order = math.floor(math.log10(val)) if val > 0 else 0
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