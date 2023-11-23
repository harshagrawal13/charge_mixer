import os
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tabulate import tabulate

pd.set_option("display.float_format", "{:.3f}".format)


class ChargeMixer:
    def __init__(
        self,
        raw_mat_info_file_path,
        out_charge_mix_file_path,
        mode,
        heat_size=50,
        furnace_size=100,
    ) -> None:
        # Ensure file paths exist
        assert os.path.exists(
            raw_mat_info_file_path
        ), f"File {raw_mat_info_file_path} does not exist"
        assert os.path.exists(
            out_charge_mix_file_path
        ), f"File {out_charge_mix_file_path} does not exist"

        self.heat_size = heat_size
        self.furnace_size = furnace_size
        self.mode = mode

        # Read into dataframes
        print("Loading dataframes...")
        self.input_df = pd.read_csv(raw_mat_info_file_path)
        self.out_req_df = pd.read_csv(out_charge_mix_file_path).fillna(0.0)

        # Preprocess the Input DataFrame
        print("Pre-processing dataframes...")
        self.preprocessing()

    def preprocessing(self):
        if self.mode == "with_existing_with_weight_constraints":
            self.test_against_existing = True
            self.weight_constraint = True
        if self.mode == "with_existing_no_weight_constraints":
            self.test_against_existing = True
            self.weight_constraint = False
        if self.mode == "vanilla_optimization":
            self.test_against_existing = False
            self.weight_constraint = False

        if self.test_against_existing and all(
            self.input_df["substd_weight(Tons)"].fillna(0) == 0
        ):
            print("Please Enter your Existing Weights in the Input File")
            return
        if (
            self.weight_constraint
            and all(self.input_df["min_weight"].fillna(0) == 0)
            and all(self.input_df["max_weight"].fillna("-") == "-")
        ):
            print("Please Enter your Min-Max Weights Constraints in the Input File")
            return

        self.non_comp_list = [
            "inputs",
            "cost_per_ton",
            "opt_cost",
            "avl_quantity",
            "total_recovery_weight",
            "min_weight",
            "max_weight",
        ]

        if self.test_against_existing:
            self.input_df = self.input_df[
                self.input_df["substd_weight(Tons)"].fillna(0) != 0
            ]
            self.non_comp_list = self.non_comp_list + ["substd_weight(Tons)"]
        else:
            self.input_df = self.input_df.drop(columns=["substd_weight(Tons)"])

        # Get the list of all columns
        cols = list(self.input_df.columns)

        # Remove Unwanted columns
        if "impurity" in cols:
            cols.remove("impurity")
        if "Fe" in cols:
            cols.remove("Fe")

        # Miscelleneous step for sorting
        for i in self.non_comp_list:
            cols.remove(i)

        cols = self.non_comp_list + cols

        # Sorting the DataFrame
        self.input_df = self.input_df[cols]

        # Preprocessing Cost per Ton and checking availability
        if self.input_df["cost_per_ton"].dtype == str:
            self.input_df["cost_per_ton"] = (
                self.input_df["cost_per_ton"].str.replace(",", "").astype(int)
            )
        if self.input_df["opt_cost"].dtype == str:
            self.input_df["opt_cost"] = (
                self.input_df["opt_cost"].str.replace(",", "").astype(int)
            )
        self.input_df = self.input_df[self.input_df["cost_per_ton"] != 0]

        # After multiplying each value by yield, the values are in unit weight.
        self.input_df.iloc[:, len(self.non_comp_list) :] = self.input_df.iloc[
            :, len(self.non_comp_list) :
        ].multiply(0.01)

        if self.input_df["total_recovery_weight"].dtype != float:
            self.input_df["total_recovery_weight"] = (
                self.input_df["total_recovery_weight"]
                .str.replace("%", "")
                .astype(float)
                * 0.01
            )

    def substandard_constraints_results(self):
        self.substd_constraints = pd.DataFrame(
            self.input_df.iloc[:, len(self.non_comp_list) :]
            .fillna(0)
            .multiply(
                self.input_df["substd_weight(Tons)"]
                / (self.input_df["substd_weight(Tons)"].sum(0)),
                axis="index",
            )
            .multiply(100)
            .sum(axis=0),
            columns=["substd_percent"],
        ).reset_index(names=["elements"])
        self.substd_constraints = self.out_req_df.merge(
            self.substd_constraints, how="right", on="elements"
        )

        self.substd_constraints["InRange"] = (
            self.substd_constraints["max"] > self.substd_constraints["substd_percent"]
        ) & (self.substd_constraints["min"] < self.substd_constraints["substd_percent"])

        self.substd_constraints = self.substd_constraints.fillna("-")

        self.substd_constraints.loc[
            self.substd_constraints["min"] == "-", "InRange"
        ] = True

    def get_optimizer_inputs(self):
        # elements composition
        elements_list = self.out_req_df["elements"].tolist()
        elements_list += ["total_recovery_weight"]

        A_ub = []
        for element in elements_list:
            A_ub.append(self.input_df[element].to_list())

        if self.weight_constraint:
            bounds = list(
                zip(self.input_df["min_weight"].fillna(0), self.input_df["max_weight"])
            )

        else:
            avl_bnds = self.input_df["avl_quantity"].to_list()
            bounds = list(map(lambda b: (0, b), avl_bnds))

        # Get all raw material names serially
        raw_mat_names = self.input_df["inputs"].tolist()

        # Get all costs for raw materials serially
        raw_mat_costs = (
            (self.input_df["cost_per_ton"] + self.input_df["opt_cost"])
        ).tolist()

        # Get output requirements: taking minimum for now
        min_percentages = self.out_req_df["min"].multiply(0.01).to_list()
        max_percentages = self.out_req_df["max"].multiply(0.01).to_list()

        min_percentages += [1.0]
        max_percentages += [1.0]

        return (
            np.array(A_ub),
            raw_mat_names,
            raw_mat_costs,
            min_percentages,
            max_percentages,
            bounds,
        )

    def relax_constraints(
        self, min_percentages: list, max_percentages: list, relax_amount=0.001
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
            bounds,
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
                bounds=bounds,
            )

            # Print the results
            if result.success:
                self.print_results(result, raw_mat_names)

                return result

            else:
                min_percentages, max_percentages = self.relax_constraints(
                    min_percentages=min_percentages, max_percentages=max_percentages
                )
                curr_try += 1

            if curr_try % 5 == 0:
                print("Current Try: ", curr_try + 1)

        print("Max tries reached! Optimization Failed. Increase number of retries")

        return None

    def substandard_test_results(self, input_mix_df):
        final_comp_df = self.input_df.merge(input_mix_df, on="inputs", how="left")
        final_comp_df = final_comp_df.iloc[:, len(self.non_comp_list) : -1].multiply(
            final_comp_df["optz_weight(Tons)"], axis=0
        )

        final_comp_df = pd.DataFrame(
            final_comp_df.sum(axis=0).multiply(100 / self.heat_size),
            columns=["optz_percent"],
        ).reset_index(names=["elements"])

        self.final_constraint_df = self.substd_constraints.merge(
            final_comp_df, how="right", on="elements"
        )

        self.final_result_df = self.input_df[self.non_comp_list]

        self.final_result_df = self.final_result_df.merge(
            input_mix_df,
            how="outer",
            on="inputs",
        )
        self.final_result_df = self.final_result_df[
            (self.final_result_df["substd_weight(Tons)"] != 0)
            | (self.final_result_df["optz_weight(Tons)"] != 0)
        ].reset_index(drop=True)

        self.final_result_df["substd_cost(Rs.)"] = self.final_result_df[
            "substd_weight(Tons)"
        ].multiply(
            (self.final_result_df["cost_per_ton"] + self.final_result_df["opt_cost"])
        )

        self.final_result_df["optimised_cost(Rs.)"] = self.final_result_df[
            "optz_weight(Tons)"
        ].multiply(
            (self.final_result_df["cost_per_ton"] + self.final_result_df["opt_cost"])
        )

        temp_lst = [
            "substd_weight(Tons)",
            "optz_weight(Tons)",
            "substd_cost(Rs.)",
            "optimised_cost(Rs.)",
        ]
        self.final_result_df.loc["Total"] = self.final_result_df[temp_lst].sum()

        self.final_result_df = self.final_result_df[["inputs"] + temp_lst]

    def optimization_results(self, input_mix_df):
        final_comp_df = self.input_df.merge(input_mix_df, on="inputs", how="left")
        final_comp_df = final_comp_df.iloc[:, len(self.non_comp_list) : -1].multiply(
            final_comp_df["optz_weight(Tons)"], axis=0
        )

        final_comp_df = pd.DataFrame(
            final_comp_df.sum(axis=0).multiply(100 / self.heat_size),
            columns=["optz_percent"],
        ).reset_index(names=["elements"])

        self.final_constraint_df = self.out_req_df.merge(
            final_comp_df, how="outer", on="elements"
        ).fillna("-")

    def print_results(self, result, raw_mat_names):
        print("Optimization Successful!")
        print(f"\nRequired Heat(Output) Size: {self.heat_size} Tons ")
        print("\nInput Mix:")

        input_mix_items = []
        for i, percentage in enumerate(result.x):
            input_mix_items.append(
                {
                    "inputs": raw_mat_names[i],
                    "optz_weight(Tons)": percentage * self.heat_size,
                }
            )
        input_mix_df = pd.DataFrame(input_mix_items)

        out_df = input_mix_df[input_mix_df["optz_weight(Tons)"] != 0].reset_index(
            drop=True
        )
        out_df = out_df.merge(self.input_df, on="inputs", how="left")
        out_df["Cost(Rs.)"] = (out_df["cost_per_ton"] + out_df["opt_cost"]).multiply(
            out_df["optz_weight(Tons)"], axis=0
        )

        out_df = out_df[["inputs", "optz_weight(Tons)", "Cost(Rs.)"]]

        out_df.loc["Total"] = out_df[["optz_weight(Tons)", "Cost(Rs.)"]].sum()

        print(
            tabulate(
                out_df.fillna("-"),
                headers="keys",
                tablefmt="psql",
                floatfmt=".3f",
            )
        )

        total_weight = input_mix_df["optz_weight(Tons)"].sum()

        if self.test_against_existing:
            self.substandard_constraints_results()
            self.substandard_test_results(input_mix_df)
        else:
            self.optimization_results(input_mix_df)

        print(f"\nTotal Cost Per Ton (Total/Heat Size): {result.fun:.2f} Rs.")

        print(f"\nFurnace Size: {self.furnace_size} Tons ")
        print(
            "Number of furnace cycles required: ",
            total_weight // self.furnace_size + 1,
        )
        print(
            f"Weight per Cycle: {total_weight / (total_weight // self.furnace_size + 1):.3f} Tons",
        )

        print("\nFinal Composition (percentage):")
        print(
            tabulate(
                self.final_constraint_df,
                headers="keys",
                tablefmt="psql",
                floatfmt=".3f",
            )
        )
        if self.test_against_existing:
            print("\nSubStandard Vs Optimised Weight Results:")
            print(
                tabulate(
                    self.final_result_df[
                        ["inputs", "substd_weight(Tons)", "optz_weight(Tons)"]
                    ].fillna("-"),
                    headers="keys",
                    tablefmt="psql",
                    floatfmt=".3f",
                )
            )

            print("\nSubStandard Vs Optimised Cost Results:")

            print(
                tabulate(
                    self.final_result_df[
                        ["inputs", "substd_cost(Rs.)", "optimised_cost(Rs.)"]
                    ].fillna("-"),
                    headers="keys",
                    tablefmt="psql",
                    floatfmt=".3f",
                )
            )
