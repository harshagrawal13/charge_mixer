import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class ChargeMixer:
    def __init__(
        self, raw_mat_info_file_path, out_charge_mix_file_path, heat_size=50
    ) -> None:
        # Ensure file paths exist
        assert os.path.exists(
            raw_mat_info_file_path
        ), f"File {raw_mat_info_file_path} does not exist"
        assert os.path.exists(
            out_charge_mix_file_path
        ), f"File {out_charge_mix_file_path} does not exist"

        self.heat_size = heat_size
        # Read into dataframes
        print("Loading dataframes...")
        self.input_df = pd.read_csv(raw_mat_info_file_path)
        self.out_req_df = pd.read_csv(out_charge_mix_file_path)

        # Preprocess the Input DataFrame
        print("Pre-processing dataframes...")
        self.preprocessing()

    def preprocessing(self):
        # Get the list of all columns
        cols = list(self.input_df.columns)

        self.non_comp_list = [
            "inputs",
            "cost_per_ton",
            "qty_avl_tons",
            "yield",
            "opt_cost",
        ]

        # Miscelleneous step for sorting
        for i in self.non_comp_list:
            cols.remove(i)

        # Remove Unwanted columns
        if "impurity" in cols:
            cols.remove("impurity")
        if "Fe" in cols:
            cols.remove("Fe")

        cols = self.non_comp_list + cols

        # Sorting the DataFrame
        self.input_df = self.input_df[cols]

        # Preprocessing Cost per Ton and checking availability
        self.input_df["cost_per_ton"] = (
            self.input_df["cost_per_ton"].str.replace(",", "").astype(int)
        )
        self.input_df = self.input_df[self.input_df["cost_per_ton"] != 0]

        # Preprocessing Yield: Percentage to Float
        if self.input_df["yield"].dtype != "float64":
            self.input_df["yield"] = (
                self.input_df["yield"].str.replace("%", "").astype(float).multiply(0.01)
            )

        # After multiplying each value by yield, the values are in unit weight.
        self.input_df.iloc[:, len(self.non_comp_list) :] = (
            self.input_df.iloc[:, len(self.non_comp_list) :]
            .multiply(self.input_df["yield"], axis=0)
            .multiply(0.01)
        )

        # Making the Fe column
        self.input_df["Fe"] = self.input_df["yield"] - self.input_df.iloc[
            :, len(self.non_comp_list) :
        ].sum(axis=1)

    def get_optimizer_inputs(self):
        # elements composition
        elements_list = self.out_req_df["elements"].tolist()

        A_ub = []
        for element in elements_list:
            A_ub.append(self.input_df[element].to_list())

        remaining_elements = list(
            set(self.input_df.columns.tolist())
            - {"yield", "inputs", "cost_per_ton", "opt_cost"}
            - set(elements_list)
        )

        # Append the remaning elements to A_ub
        A_ub.append(self.input_df[remaining_elements].sum(axis=1).to_list())

        # Get all raw material names serially
        raw_mat_names = self.input_df["inputs"].tolist()

        # Get all costs for raw materials serially
        raw_mat_costs = (
            self.input_df["cost_per_ton"] + self.input_df["opt_cost"]
        ).tolist()

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
                        "weight(ton)": percentage * self.heat_size,
                    }
                )
        input_mix_df = pd.DataFrame(input_mix_items)
        print(input_mix_df)

        print("\n Input Heat Size: ", self.heat_size)
        print("\n Tried Input Weight: ", input_mix_df["weight(ton)"].sum())
        print("\n Total Cost Per Heat: ", result.fun * self.heat_size)
        print("\n Total Cost Per Ton: ", result.fun)

        final_comp_df = self.input_df.join(
            input_mix_df.set_index("inputs"), on="inputs", how="inner"
        )
        final_comp_df = final_comp_df.iloc[:, len(self.non_comp_list) : -1].multiply(
            final_comp_df["weight(ton)"], axis=0
        )
        print("\nFinal Composition (percentage):")
        print(final_comp_df.sum(axis=0).multiply(100 / self.heat_size))
        print(
            f"Total: {final_comp_df.sum(axis=0).multiply(100 / self.heat_size).sum()}"
        )
