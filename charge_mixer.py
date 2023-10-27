import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class ChargeMixer:
    def __init__(
        self, raw_mat_info_file_path, raw_mat_cost_file_path, out_charge_mix_file_path
    ) -> None:
        # Ensure file paths exist
        assert os.path.exists(
            raw_mat_info_file_path
        ), f"File {raw_mat_info_file_path} does not exist"
        assert os.path.exists(
            raw_mat_cost_file_path
        ), f"File {raw_mat_cost_file_path} does not exist"
        assert os.path.exists(
            out_charge_mix_file_path
        ), f"File {out_charge_mix_file_path} does not exist"

        # Read into dataframes
        print("Loading dataframes...")
        self.charge_mix_df = pd.read_csv(raw_mat_info_file_path)
        self.input_cost_df = pd.read_csv(raw_mat_cost_file_path)
        self.out_req_df = pd.read_csv(out_charge_mix_file_path)

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

        remaining_elements = list(
            set(self.merged.columns.tolist())
            - {"impurity", "yield", "inputs", "cost_per_ton"}
            - set(elements_list)
        )

        # Append the remaning elements to A_ub
        A_ub.append(self.merged[remaining_elements].sum(axis=1).to_list())

        # Get all raw material names serially
        raw_mat_names = self.merged["inputs"].tolist()

        # Get all costs for raw materials serially
        raw_mat_costs = self.merged["cost_per_ton"].tolist()

        # Get output requirements: taking minimum for now
        min_percentages = self.out_req_df["min"].multiply(0.01).to_list()
        max_percentages = self.out_req_df["max"].multiply(0.01).to_list()

        others_min = 1 - sum(max_percentages)
        others_max = 1 - sum(min_percentages)

        min_percentages.append(others_min)
        max_percentages.append(others_max)

        return (
            np.array(A_ub),
            raw_mat_names,
            raw_mat_costs,
            min_percentages,
            max_percentages,
        )

    def relax_constraints(
        self, min_percentages: list, max_percentages: list, relax_amount=2.0
    ):
        """Relax constraint for total weight by the set relax amount

        Args:
            min_percentages (list): Min Percentages.
            max_percentages (list): Max Percentages
            relax_amount (float, optional): relax amount float. Defaults to 2.0.
        """
        min_percentages[-1] = min_percentages[-1] - relax_amount
        max_percentages[-1] = max_percentages[-1] + relax_amount
        return min_percentages, max_percentages

    def run_optimization(self, retries=10):
        curr_try = 0
        (
            A_ub,
            raw_mat_names,
            raw_mat_costs,
            min_percentages,
            max_percentages,
        ) = self.get_optimizer_inputs()

        while curr_try <= retries:
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

            if curr_try % 5 == 0:
                print("Current Try: ", curr_try + 1)

            # Print the results
            if result.success:
                print("Current Try: ", curr_try + 1)
                self.print_results(result, raw_mat_names)
                return result

            else:
                min_percentages, max_percentages = self.relax_constraints(
                    min_percentages=min_percentages, max_percentages=max_percentages
                )
                curr_try += 1

        print("Max tries reached! Optimization Failed. Increase number of retries")
        return None

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
        print("\nFinal Composition (unit weight):")
        print(final_comp_df.sum(axis=0).multiply(100))
        print(f"Total: {final_comp_df.sum(axis=0).multiply(100).sum()}")
