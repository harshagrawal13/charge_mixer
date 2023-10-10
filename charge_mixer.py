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
        # Merge both the DB to ensure we only take the rows needed.
        self.merged = self.input_cost_df.merge(
            self.charge_mix_df, on="inputs", how="inner"
        )

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
        self.charge_mix_df.loc[:, "C":"impurity"] = (
            self.charge_mix_df.loc[:, "C":"impurity"]
            .multiply(self.charge_mix_df["yield"], axis=0)
            .multiply(0.01)
        )

    def preprocess_costs_df(self):
        self.input_cost_df = pd.read_csv("data/input_costs.csv")
        # Remove available qty column for now.
        self.input_cost_df.drop(["qty_avl_tons"], axis=1, inplace=True)
        self.input_cost_df["cost_per_ton"] = (
            self.input_cost_df["cost_per_ton"].str.replace(",", "").astype(int)
        )
        # Drop rows with cost_per_ton = 0. assuming raw material isn't available.
        self.input_cost_df = self.input_cost_df[self.input_cost_df["cost_per_ton"] != 0]

    def get_optimizer_inputs(self):
        # elements composition
        elements_list = self.out_req_df["elements"].tolist()

        A_ub = []
        for element in elements_list:
            A_ub.append(self.merged[element].to_list())

        # Get all raw material names serially
        raw_mat_names = self.merged["inputs"].tolist()

        # Get all costs for raw materials serially
        raw_mat_costs = self.merged["cost_per_ton"].tolist()

        # Get output requirements: taking minimum for now
        min_percentages = self.out_req_df["min"].multiply(0.01).to_list()
        max_percentages = self.out_req_df["max"].multiply(0.01).to_list()

        return (
            np.array(A_ub),
            elements_list,
            raw_mat_names,
            raw_mat_costs,
            min_percentages,
            max_percentages,
        )

    def run_optimization(self):
        (
            A_ub,
            elements_list,
            raw_mat_names,
            raw_mat_costs,
            min_percentages,
            max_percentages,
        ) = self.get_optimizer_inputs()

        costs = raw_mat_costs
        A_min = -A_ub
        b_min = -np.array(min_percentages)

        A_max = A_ub
        b_max = np.array(max_percentages)

        result = linprog(
            costs,
            A_ub=np.vstack([A_min, A_max]),
            b_ub=np.hstack([b_min, b_max]),
            bounds=(0, None),
        )

        # Print the results
        if result.success:
            self.print_results(result, raw_mat_names)
        return result

    def print_results(self, result, raw_mat_names):
        print("Optimization Successful!")
        print("Input Mix:")

        input_mix_items = []
        for i, percentage in enumerate(result.x):
            if percentage > 0:
                input_mix_items.append(
                    {
                        "inputs": raw_mat_names[i],
                        "percentage(%)": percentage * 100,
                    }
                )
        input_mix_df = pd.DataFrame(input_mix_items)
        print(input_mix_df)

        print("\n Total Cost: ", result.fun)
        final_comp_df = self.merged.join(
            input_mix_df.set_index("inputs"), on="inputs", how="inner"
        )
        final_comp_df = final_comp_df.loc[:, "C":"Fe"].multiply(
            final_comp_df["percentage(%)"] / 100, axis=0
        )
        print("\nFinal Composition (%):")
        print(final_comp_df.sum(axis=0).multiply(100))
