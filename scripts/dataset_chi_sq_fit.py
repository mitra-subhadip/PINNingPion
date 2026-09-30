import numpy as np
import pandas as pd
import torch

from helper import load_model, set_torch_device


def summarize_by_source(df, chi2_points, region):
    result = pd.DataFrame({
        "Region": region,
        "Source": df["Source"],
        "chi2_point": chi2_points.detach().cpu().numpy(),
    })

    result = (
        result.groupby(["Region", "Source"], sort=False)["chi2_point"]
        .agg(chi2="sum", N="count")
        .reset_index()
    )

    result["chi2/N"] = result["chi2"] / result["N"]
    return result


def calculate_chi2_by_source(model, df_space, df_time, device):
    model.eval()
    summaries = []

    # Spacelike
    if df_space is not None:
        q2 = torch.tensor(
            -df_space["q2"].values,
            dtype=torch.float32,
            device=device,
        )

        F_true_np = np.sqrt(df_space["Fpi_sq"].values)
        err_np = df_space["error"].values / (2.0 * F_true_np + 1e-8)

        F_true = torch.tensor(
            F_true_np,
            dtype=torch.float32,
            device=device,
        )
        error = torch.tensor(
            err_np,
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            F_pred, _ = model.forward(q2, torch.zeros_like(q2))
            chi2_points = ((F_pred.reshape(-1) - F_true) / error) ** 2

        summaries.append(
            summarize_by_source(df_space, chi2_points, "Spacelike")
        )

    # Timelike
    if df_time is not None:
        q2 = torch.tensor(
            df_time["q2"].values,
            dtype=torch.float32,
            device=device,
        )

        # Isospin flag of the dataset (I=1 for tau, I=0 for e+e-)
        I = torch.tensor(
            df_time["I"].values,
            dtype=torch.float32,
            device=device,
        )

        F_sq_true = torch.tensor(
            df_time["Fpi_sq"].values,
            dtype=torch.float32,
            device=device,
        )
        error = torch.tensor(
            df_time["error"].values,
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            u_pred, v_pred = model.forward_mixed(q2, 1 - I)
            F_sq_pred = u_pred.reshape(-1) ** 2 + v_pred.reshape(-1) ** 2
            chi2_points = ((F_sq_pred - F_sq_true) / error) ** 2

        summaries.append(
            summarize_by_source(df_time, chi2_points, "Timelike")
        )

    source_results = pd.concat(summaries, ignore_index=True)

    region_results = (
        source_results.groupby("Region", sort=False)
        .agg(chi2=("chi2", "sum"), N=("N", "sum"))
        .reset_index()
    )
    region_results["chi2/N"] = (
        region_results["chi2"] / region_results["N"]
    )

    total_chi2 = region_results["chi2"].sum()
    total_n = region_results["N"].sum()

    region_results.loc[len(region_results)] = {
        "Region": "Total",
        "chi2": total_chi2,
        "N": total_n,
        "chi2/N": total_chi2 / total_n,
    }

    print("\nCHI-SQUARED BY SOURCE")
    print(
        source_results.to_string(
            index=False,
            formatters={
                "chi2": "{:.3f}".format,
                "chi2/N": "{:.3f}".format,
            },
        )
    )

    print("\nGLOBAL CHI-SQUARED")
    print(
        region_results.to_string(
            index=False,
            formatters={
                "chi2": "{:.3f}".format,
                "chi2/N": "{:.3f}".format,
            },
        )
    )

    return source_results, region_results


if __name__ == "__main__":
    df_space = pd.read_csv("../dataset/spacelike_dataset.csv")
    df_time = pd.read_csv("../dataset/timelike_dataset.csv")

    device = set_torch_device()
    model = load_model(device)

    source_chi2, global_chi2 = calculate_chi2_by_source(
        model,
        df_space,
        df_time,
        device,
    )

    source_chi2.to_csv("chi2_by_source.csv", index=False)
