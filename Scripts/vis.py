"""
Usage:
python scripts/vis.py -name Data/AP01
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-name", required=True,help="Path to participant folder")
args = parser.parse_args()
participant_path = args.name
VIS_DIR = "Visualizations"

participant = os.path.basename(participant_path)
dfSP02 = pd.read_csv(os.path.join(participant_path, "SPO2.txt"),sep=';', skiprows=7, names=["Time","Value"])
dfFlow = pd.read_csv(os.path.join(participant_path, "Flow.txt"),sep=';', skiprows=7, names=["Time","Value"])
dfThorac = pd.read_csv(os.path.join(participant_path, "Thorac.txt"),sep=';', skiprows=7, names=["Time","Value"])
dfFlowEvents = pd.read_csv(os.path.join(participant_path, "Flow Events.txt"),sep=';', skiprows=5,names=["Range","Duration","Event","Stage"])

dfSP02["Time"] = pd.to_datetime(dfSP02["Time"],format="%d.%m.%Y %H:%M:%S,%f")
dfFlow["Time"] = pd.to_datetime(dfFlow["Time"],format="%d.%m.%Y %H:%M:%S,%f")
dfThorac["Time"] = pd.to_datetime(dfThorac["Time"],format="%d.%m.%Y %H:%M:%S,%f")

dfFlowEvents[["Start","End"]] = dfFlowEvents["Range"].str.split("-", expand=True)
dfFlowEvents["Date"] = dfFlowEvents["Start"].str.split().str[0]
dfFlowEvents["Start"] = pd.to_datetime(dfFlowEvents["Start"],format="%d.%m.%Y %H:%M:%S,%f")
dfFlowEvents["End"] = pd.to_datetime(dfFlowEvents["Date"] + " " + dfFlowEvents["End"],format="%d.%m.%Y %H:%M:%S,%f")

dfSP02.set_index("Time",inplace = True)
dfFlow.set_index("Time",inplace = True)
dfThorac.set_index("Time",inplace = True)

def clean_signal(df, signal_type, column="Value"):

    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")

    if signal_type == "spo2":

        df.loc[(df[column] < 70) | (df[column] > 100), column] = np.nan

        jump = df[column].diff().abs()
        df.loc[jump > 4, column] = np.nan

        df[column] = df[column].interpolate("time", limit=20)

        df[column] = df[column].rolling(8, center=True, min_periods=1).mean()

    elif signal_type == "flow":

        low = df[column].quantile(0.01)
        high = df[column].quantile(0.99)
        df[column] = df[column].clip(low, high)

        baseline = df[column].rolling(320, center=True, min_periods=1).mean()
        df[column] = df[column] - baseline

        df[column] = df[column].rolling(5, center=True, min_periods=1).mean()

    elif signal_type == "thorac":

        z = (df[column] - df[column].mean()) / df[column].std()
        df.loc[np.abs(z) > 5, column] = np.nan

        df[column] = df[column].interpolate(limit=50)

        baseline = df[column].rolling(480, center=True, min_periods=1).mean()
        df[column] = df[column] - baseline

    return df

dfSP02   = clean_signal(dfSP02, "spo2")
dfFlow   = clean_signal(dfFlow, "flow")
dfThorac = clean_signal(dfThorac, "thorac")

target_freq = "31.25ms"

dfSP02 = (dfSP02.resample(target_freq).interpolate(method="time"))
dfFlow = dfFlow.resample(target_freq).mean().interpolate("time")
dfThorac = dfThorac.resample(target_freq).mean().interpolate("time")

start = max(dfSP02.index[0],dfFlow.index[0],dfThorac.index[0])
end = min(dfSP02.index[-1],dfFlow.index[-1],dfThorac.index[-1])

window = pd.Timedelta(minutes=5)
ranges = pd.date_range(start=start,end=end,freq=window)

pdf_path = os.path.join(VIS_DIR, f"{participant}_visualization.pdf")
os.makedirs(VIS_DIR, exist_ok=True)

print(f"Generating visualization for {participant}...")
with PdfPages(pdf_path) as pdf:

    for start in ranges:

        end = start + window

        dfSP02_5min = dfSP02.loc[start:end]
        dfFlow_5min = dfFlow.loc[start:end]
        dfThorac_5min = dfThorac.loc[start:end]
        events_5min = dfFlowEvents[(dfFlowEvents["Start"] <= end) &(dfFlowEvents["End"] >= start)]

        fig, ax = plt.subplots(3, 1,figsize=(18,8),sharex=True,constrained_layout=True)

        fig.patch.set_facecolor("white")
        fig.patch.set_edgecolor("black")
        fig.patch.set_linewidth(1.2)

        ax[0].plot(dfFlow_5min.index, dfFlow_5min["Value"],color='tab:blue',label = "Nasal Flow")
        ax[0].legend(loc = "upper right")
        ax[0].set_ylabel("Nasal Flow (L/min)")
        for _, event in events_5min.iterrows():
          start_e = max(event["Start"], start)
          end_e   = min(event["End"], end)
          ax[0].axvspan(start_e,end_e,color="palegoldenrod",alpha=0.6,zorder=0)
          mid_time = start_e + (end_e - start_e) / 2
          y_pos = ax[0].get_ylim()[1] * 0.85
          ax[0].text(mid_time,y_pos,event["Event"],ha="center",va="center",fontsize=9,color="black",bbox=dict(facecolor="white",alpha=0.6,edgecolor="none",boxstyle="round,pad=0.2"))

        ax[1].plot(dfThorac_5min.index, dfThorac_5min["Value"],color='tab:orange',label = "Thoracic/Abdominal Resp.")
        ax[1].legend(loc = "upper right")
        ax[1].set_ylabel("Resp. Amplitude")

        ax[2].plot(dfSP02_5min.index, dfSP02_5min["Value"],color='grey',label = "SpO₂")
        ax[2].legend(loc = "upper right")
        ax[2].set_ylabel("SpO2 (%)")

        ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M:%S"))
        ax[2].xaxis.set_major_locator(mdates.SecondLocator(interval=5))
        plt.setp(ax[2].get_xticklabels(), rotation=90)
        fig.supxlabel("Time")

        for a in ax:
            a.grid(True, color="lightgray")
            for spine in a.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1)

        fig.suptitle(f"{participant} - {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}",fontsize=14)

        pdf.savefig(fig)
        plt.close(fig)
print(f"Saved to {pdf_path}")
