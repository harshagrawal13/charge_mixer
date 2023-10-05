import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class ChargeMixer:
    def __init__(self, charge_mix_file_path, raw_mats_file_path, out_file_path) -> None:
        # Ensure file paths exist
        assert os.path.exists(
            charge_mix_file_path
        ), f"File {charge_mix_file_path} does not exist"
        assert os.path.exists(
            raw_mats_file_path
        ), f"File {raw_mats_file_path} does not exist"
        assert os.path.exists(out_file_path), f"File {out_file_path} does not exist"

        # Read into dataframes
        print("Loading dataframes...")
        self.charge_mix_df = pd.read_csv(charge_mix_file_path)
        self.input_cost_df = pd.read_csv(raw_mats_file_path)
        self.out_req_df = pd.read_csv(out_file_path)

        # Preprocess dataframes
        print("Pre-processing dataframes...")
        self.preprocess_charge_mix_df()
        self.preprocess_costs_df()
        self.preprocess_out_req_df()

    def preprocess_charge_mix_df(self):
        self.charge_mix_df.fillna(0, inplace=True)
        # Convert yield to percentage. eg: 95% -> 0.95
        self.charge_mix_df["yield"] = (
            self.charge_mix_df["yield"]
            .str.replace("%", "")
            .astype(float)
            .multiply(0.01)
        )
        # After multiplying each value by yield, the values are in unit weight.
        self.charge_mix_df.loc[:, "C":"impurity"] = self.charge_mix_df.loc[
            :, "C":"impurity"
        ].multiply(self.charge_mix_df["yield"], axis=0)

    def preprocess_costs_df(self):
        self.input_cost_df = pd.read_csv("data/input_costs.csv")
        # Remove available qty column for now.
        self.input_cost_df.drop(["qty_avl_tons"], axis=1, inplace=True)
        self.input_cost_df["cost_per_ton"] = (
            self.input_cost_df["cost_per_ton"].str.replace(",", "").astype(int)
        )
        # Drop rows with cost_per_ton = 0. assuming raw material isn't available.
        self.input_cost_df = self.input_cost_df[self.input_cost_df["cost_per_ton"] != 0]

    def preprocess_out_req_df(self):
        self.out_req_df = pd.read_csv("data/eg_out.csv")
        self.out_req_df.fillna(0, inplace=True)
        fe_req = {
            "elements": "Fe",
            "min": 100 - self.out_req_df["min"].sum(),
            "max": 100 - self.out_req_df["max"].sum(),
        }
        self.out_req_df.loc[len(self.out_req_df)] = fe_req

    def get_optimizer_inputs(self):
        # Merge both the DB to ensure we only take the rows needed.
        merged = self.input_cost_df.merge(self.charge_mix_df, on="inputs", how="inner")

        # Get inputs about element compositions
        df_inps_to_list = []
        for col in merged.loc[:, "C":"impurity"].columns:
            df_inps_to_list.append(merged[col].tolist())

        # Get all element names serially
        elements_list = merged.loc[:, "C":"impurity"].columns.tolist()

        # Get all raw material names serially
        raw_mat_names = merged["inputs"].tolist()

        # Get all costs for raw materials serially
        raw_mat_costs = merged["cost_per_ton"].tolist()

        # Get output requirements: taking minimum for now
        all_outs = self.out_req_df["elements"].tolist()
        min_percentages = []
        max_percentages = []
        for item in elements_list:
            if item in all_outs:
                min_percentages.append(
                    self.out_req_df.loc[
                        self.out_req_df["elements"] == item, "min"
                    ].iloc[0]
                )
                max_percentages.append(
                    self.out_req_df.loc[
                        self.out_req_df["elements"] == item, "max"
                    ].iloc[0]
                )
            else:
                min_percentages.append(0)
                max_percentages.append(0)

        return (
            np.array(df_inps_to_list),
            elements_list,
            raw_mat_names,
            raw_mat_costs,
            min_percentages,
            max_percentages,
        )
