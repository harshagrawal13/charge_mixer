import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from tabulate import tabulate

pd.set_option("display.float_format", "{:.3f}".format)


class ChargeMixer:
    def __init__(
        self,
        input_file_path,
        out_file_path,
    ) -> None:
        # Ensure file paths exist
        self.out_results = {}
        self.out_file_path = out_file_path
        self.input_file_path = input_file_path

    def check_input_file(self):

        if not os.path.exists(self.input_file_path):
            self.out_results["error"] = f"File {self.input_file_path} does not exist"
            return

        with open(self.input_file_path, "r") as outfile:
            data = json.load(outfile)
        self.data = data

        modes = [
            "with_existing_with_weight_constraints",
            "with_existing_no_weight_constraints",
            "vanilla_optimization",
        ]

        if self.data["mode"] not in modes:
            self.out_results["error"] = "Please enter a valid mode"
            return

        # Read into dataframes
        print("Loading dataframes...")
        self.input_df = pd.DataFrame.from_records(self.data["raw_mat_info"])
        self.out_req_df = pd.DataFrame.from_records(self.data["out_charge_mix"]).fillna(
            {"min": 0.0, "max": 100.0}
        )

        # Preprocess the Input DataFrame
        print("Pre-processing dataframes...")
        self.preprocessing()

    def preprocessing(self):

        self.furnace_size = self.data["furnace_size"]
        self.mode = self.data["mode"]
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
            self.input_df["substd_weight_tons"].fillna(0) == 0
        ):
            print("Please Enter your Existing Weights in the Input File")
            self.out_results["error"] = (
                "Please Enter your Existing Weights in the Input File"
            )
            return

        if (
            self.weight_constraint
            and all(self.input_df["min_weight"].fillna(0) == 0)
            and all(self.input_df["max_weight"].fillna("-") == "-")
        ):
            print("Please Enter your Min-Max Weights Constraints in the Input File")
            self.out_results["error"] = (
                "Please Enter your Min-Max Weights Constraints in the Input File"
            )
            return

        self.non_comp_list = [
            "input_name",
            "cost_per_ton",
            "avl_quantity",
            "total_recovery_weight_percent",
        ]

        if self.test_against_existing:
            self.input_df = self.input_df[
                self.input_df["substd_weight_tons"].fillna(0) != 0
            ]
            self.non_comp_list = self.non_comp_list + ["substd_weight_tons"]
            print(
                f"Furnace Size: {self.furnace_size} -> {self.input_df['substd_weight_tons'].sum(0):.2f}"
            )
            self.furnace_size = self.input_df["substd_weight_tons"].sum(0)

        else:
            self.input_df = self.input_df.drop(columns=["substd_weight_tons"])

        if self.weight_constraint:
            self.input_df = self.input_df.fillna({"min_weight": 0})
            self.non_comp_list = self.non_comp_list + ["min_weight", "max_weight"]

        else:
            self.input_df = self.input_df.drop(columns=["min_weight", "max_weight"])

        self.input_df = self.input_df[self.input_df["cost_per_ton"] != 0]
        self.input_df["total_recovery_weight_percent"] = (
            self.input_df["total_recovery_weight_percent"].astype(float) * 0.01
        )

        # Get the list of all columns
        cols = list(self.input_df.columns)

        # Miscelleneous step for sorting
        for i in self.non_comp_list:
            cols.remove(i)

        cols = self.non_comp_list + cols

        # Sorting the DataFrame
        self.input_df = self.input_df[cols]

        # After multiplying each value by yield, the values are in unit weight.
        self.input_df.iloc[:, len(self.non_comp_list) :] = self.input_df.iloc[
            :, len(self.non_comp_list) :
        ].multiply(0.01)

    def get_optimizer_inputs(self):
        # elements composition
        elements_list = self.out_req_df["elements"].tolist()
        elements_list += ["total_recovery_weight_percent"]

        A_ub = []
        for element in elements_list:
            A_ub.append(self.input_df[element].to_list())

        if self.weight_constraint:
            bounds = list(
                zip(
                    self.input_df["min_weight"].multiply(1 / self.furnace_size),
                    self.input_df["max_weight"].multiply(1 / self.furnace_size),
                )
            )

        else:
            avl_bnds = self.input_df["avl_quantity"].to_list()
            bounds = list(map(lambda b: (0, b), avl_bnds))

        # Get all raw material names serially
        raw_mat_names = self.input_df["input_name"].tolist()

        # Get all costs for raw materials serially
        raw_mat_costs = (self.input_df["cost_per_ton"]).tolist()

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

    def run_optimization(self):

        self.check_input_file()
        try:
            (
                A_ub,
                raw_mat_names,
                raw_mat_costs,
                min_percentages,
                max_percentages,
                bounds,
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
                bounds=bounds,
            )
            self.get_results(result, raw_mat_names)

        except:
            if len(self.out_results) == 0:
                self.out_results["error"] = (
                    "Please Check the Input File and Columns of Data"
                )

        with open(self.out_file_path, "w") as outfile:
            json.dump(self.out_results, outfile, indent=4)

    def substandard_test_results(self, input_mix_df):
        self.substd_constraints = pd.DataFrame(
            self.input_df.iloc[:, len(self.non_comp_list) :]
            .fillna(0)
            .multiply(
                (
                    self.input_df["substd_weight_tons"]
                    * self.input_df["total_recovery_weight_percent"]
                )
                / (
                    (
                        self.input_df["substd_weight_tons"]
                        * self.input_df["total_recovery_weight_percent"]
                    ).sum(0)
                ),
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

        final_comp_df = self.input_df.merge(input_mix_df, on="input_name", how="left")
        final_comp_df = final_comp_df.iloc[:, len(self.non_comp_list) : -1].multiply(
            final_comp_df["optz_weight_tons"], axis=0
        )

        final_comp_df = pd.DataFrame(
            final_comp_df.sum(axis=0).multiply(100),
            columns=["optz_percent"],
        ).reset_index(names=["elements"])

        self.final_constraint_df = self.substd_constraints.merge(
            final_comp_df, how="right", on="elements"
        )

        self.final_result_df = self.input_df[self.non_comp_list]

        self.final_result_df = self.final_result_df.merge(
            input_mix_df,
            how="outer",
            on="input_name",
        )
        self.final_result_df = self.final_result_df[
            (self.final_result_df["substd_weight_tons"] != 0)
            | (self.final_result_df["optz_weight_tons"] != 0)
        ].reset_index(drop=True)

        self.final_result_df["optz_weight_tons"] = (
            self.final_result_df["optz_weight_tons"]
            .multiply(1 / self.final_result_df["optz_weight_tons"].sum())
            .multiply(self.furnace_size)
        )

        self.final_result_df["substd_cost_rupees"] = self.final_result_df[
            "substd_weight_tons"
        ].multiply((self.final_result_df["cost_per_ton"]))

        self.final_result_df["optimised_cost_rupees"] = self.final_result_df[
            "optz_weight_tons"
        ].multiply((self.final_result_df["cost_per_ton"]))

        temp_lst = [
            "substd_weight_tons",
            "optz_weight_tons",
            "substd_cost_rupees",
            "optimised_cost_rupees",
        ]
        self.final_result_df.loc["Total"] = self.final_result_df[temp_lst].sum()

        self.final_result_df = self.final_result_df[["input_name"] + temp_lst]

    def optimization_results(self, input_mix_df):
        final_comp_df = self.input_df.merge(input_mix_df, on="input_name", how="left")
        final_comp_df = final_comp_df.iloc[:, len(self.non_comp_list) : -1].multiply(
            final_comp_df["optz_weight_tons"], axis=0
        )

        final_comp_df = pd.DataFrame(
            final_comp_df.sum(axis=0).multiply(100),
            columns=["optz_percent"],
        ).reset_index(names=["elements"])

        self.final_constraint_df = self.out_req_df.merge(
            final_comp_df, how="outer", on="elements"
        )

    def get_results(
        self,
        result,
        raw_mat_names,
    ):
        if result.success:
            input_mix_items = []
            for i, percentage in enumerate(result.x):
                input_mix_items.append(
                    {
                        "input_name": raw_mat_names[i],
                        "optz_weight_tons": percentage,
                    }
                )
            input_mix_df = pd.DataFrame(input_mix_items)

            self.out_df = input_mix_df[
                input_mix_df["optz_weight_tons"] != 0
            ].reset_index(drop=True)
            self.out_df["optz_weight_tons"] = (
                self.out_df["optz_weight_tons"]
                .multiply(1 / self.out_df["optz_weight_tons"].sum())
                .multiply(self.furnace_size)
            )
            self.out_df = self.out_df.merge(self.input_df, on="input_name", how="left")
            self.out_df["cost_rupees"] = (self.out_df["cost_per_ton"]).multiply(
                self.out_df["optz_weight_tons"], axis=0
            )

            self.out_df = self.out_df[["input_name", "optz_weight_tons", "cost_rupees"]]
            self.out_df.loc["Total"] = self.out_df[
                ["optz_weight_tons", "cost_rupees"]
            ].sum()

            laddle_size = self.furnace_size * (
                1 / (input_mix_df["optz_weight_tons"].sum())
            )

            self.out_results["furnace_size"] = round(self.furnace_size, 3)
            self.out_results["laddle_size"] = round(laddle_size, 3)
            self.out_results["total_cost_per_ton"] = round(result.fun, 2)
            self.out_results["optimized_input_mix"] = (
                self.out_df.iloc[:-1, :].round(3).to_dict(orient="records")
            )

            if self.test_against_existing:
                self.substandard_test_results(input_mix_df)
                savings = (
                    self.final_result_df["substd_cost_rupees"]["Total"]
                    - self.final_result_df["optimised_cost_rupees"]["Total"]
                )

                self.out_results["final_composition"] = (
                    self.final_result_df.iloc[:-1, :].round(3).to_dict(orient="records")
                )

                if savings >= 0:
                    self.out_results["substandard_optimisation"] = True
                else:
                    self.out_results["substandard_optimisation"] = False

                substd_cost_per_ton_output = (
                    self.final_result_df["substd_cost_rupees"]["Total"]
                ) / (
                    (
                        self.input_df["substd_weight_tons"]
                        * self.input_df["total_recovery_weight_percent"]
                    ).sum(0)
                )

                self.out_results["substandard_cost_per_ton_output"] = round(
                    substd_cost_per_ton_output, 2
                )
                self.out_results["savings"] = round(savings, 2)

            else:
                self.optimization_results(input_mix_df)

            self.out_results["final_element_composition"] = (
                self.final_constraint_df.iloc[:-1, :].round(3).to_dict(orient="records")
            )

        else:
            self.out_results["error"] = "Optimization Failed. Change the Inputs or Mode"

    def print_results(self):

        if len(self.out_results) == 0:
            print("Optimization Failed. Change the Inputs or Mode")
            return None

        print("Optimization Successful!")
        print(f"\nFurnace(Input) Size: {self.out_results['furnace_size']} Tons ")
        print("\nInput Mix:")
        print(
            tabulate(
                self.out_df.fillna("-"),
                headers="keys",
                tablefmt="psql",
                floatfmt=".3f",
            )
        )

        print(f"\nLaddle(Output) Size: {self.out_results['laddle_size']} Tons ")
        print(
            f"Total Cost Per Ton (Total/Laddle Size): {self.out_results['total_cost_per_ton']} Rs."
        )

        print("\nFinal Composition (percentage):")
        print(
            tabulate(
                self.final_constraint_df.round(3)
                .replace(to_replace=0.0, value=np.nan)
                .fillna("-"),
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
                        ["input_name", "substd_weight_tons", "optz_weight_tons"]
                    ]
                    .round(3)
                    .replace(to_replace=0.0, value=np.nan)
                    .fillna("-"),
                    headers="keys",
                    tablefmt="psql",
                    floatfmt=".3f",
                )
            )

            print("\nSubStandard Vs Optimised Cost Results:")
            print(
                tabulate(
                    self.final_result_df[
                        ["input_name", "substd_cost_rupees", "optimised_cost_rupees"]
                    ]
                    .round(3)
                    .replace(to_replace=0.0, value=np.nan)
                    .fillna("-"),
                    headers="keys",
                    tablefmt="psql",
                    floatfmt=".3f",
                )
            )
            print(f"\n Total Savings: {self.out_results['savings']} Rs.")
