
import importlib_resources
import json
import pandas as pd
import altair as alt
from datetime import timedelta
dt = timedelta(days=180)

data_path = importlib_resources.files("taranis.data")


with open(data_path / "amd.json", encoding="utf-8") as file:
    amd = json.load(file)

with open(data_path / "nvidia.json", encoding="utf-8") as file:
    nvidia = json.load(file)

with open(data_path / "intel.json", encoding="utf-8") as file:
    intel = json.load(file)


with open(data_path / "tenstorrent.json", encoding="utf-8") as file:
    tens = json.load(file)


tens = []

data = pd.DataFrame(pd.json_normalize(nvidia + amd + intel + tens, sep='_').to_dict(orient='records'))
data['release'] = pd.to_datetime(data['release'], format='%d/%m/%Y', errors='ignore')


flops = {
    "TOPS": 1,
    'TFLOPS': 1,
    'GFLOPS': 1000,
}

transistors = {
    "M": 15300
}

surface = {
    "mm2": 610
}

def split_unit(name, format=None):
    data[[name, f'{name}_unit']] = data[name].str.split(' ', n=1, expand=True)
    data[name] = data[name].astype(float, errors='ignore')

    if format:
        multipliers = data[f'{name}_unit'].map(format)                 
        data[name] = data[name] / multipliers
        

start = data['release'].min()
end = data['release'].max()

split_unit('performance_int8', flops)
split_unit('performance_fp16', flops)
split_unit('performance_fp32', flops)
split_unit('performance_fp64', flops)
split_unit('performance_tf32', flops)
split_unit('power_TDP')
# split_unit('transitors', transistors)
# split_unit('die', surface)
split_unit('performance_fp16_nvidia', flops)
split_unit('performance_bf16_nvidia', flops)
split_unit('performance_tf32_nvidia', flops)
split_unit('performance_fp32_nvidia', flops)
split_unit('performance_fp64_nvidia', flops)

denom = 'performance_fp32_nvidia'

data['ratio_fp64'] = (data['performance_fp64_nvidia'] / data[denom])
data['ratio_fp16'] = (data['performance_fp16_nvidia'] / data[denom])
data['ratio_tf32'] = (data['performance_tf32_nvidia'] / data[denom])

denom = 'performance_fp32'

data['ratio_fp64'].fillna(data['performance_fp64'] / data[denom], inplace=True)
data['ratio_fp16'].fillna(data['performance_fp16'] / data[denom], inplace=True)
data['ratio_tf32'].fillna(data['performance_tf32'] / data[denom], inplace=True)

print(data[["name", 'ratio_fp64', 'ratio_tf32', 'ratio_fp16', 'performance_fp32', 'transitors', 'die']])


def graphs():
    nearest = alt.selection(
        name='p',
        type='single', 
        nearest=True, 
        on='mouseover',
        fields=['release', 'performance_fp16']
    )

    def perf_evol(measure='performance_fp16'):
        base = alt.Chart(data, title=measure)

        mn = data[measure].min()
        mx = data[measure].max()

        x = alt.X('release:T', scale=alt.Scale(domain=(start - dt, end + dt)))
        
        points = base.mark_point().encode(
            x=x,
            y=alt.Y(f'{measure}:Q', scale=alt.Scale(domain=(mn * 0.9, mx * 1.1))),
            color='vendor'
        ).add_selection(
            nearest
        )
        
        txt = base.mark_text(
            dx=0, 
            dy=0,
            xOffset=5,
            yOffset=-5,
            align="left",
            baseline="middle",
            font="monospace",
            limit=50,
            angle=360 - 45,
        ).encode(
            x=x,
            y=alt.Y(f'{measure}:Q'),
            text=alt.condition(nearest, 'name:N', alt.value(' ')),
            color='vendor',
        )
        
        return (points + txt)


    def perf_per_watt(measure='performance_fp16'):
        base = alt.Chart(data, title="perf per watt")
        x = alt.X('release:T', scale=alt.Scale(domain=(start - dt, end + dt)))
        _, dtype = measure.split('_')

        points = base.mark_point().encode(
            x=x,
            y=f'perf_per_watt_{dtype}:Q',
            color='vendor'
        ).add_selection(
            nearest
        )

        txt = base.mark_text(
            dx=0, 
            dy=0,
            xOffset=5,
            yOffset=-5,
            align="left",
            baseline="middle",
            font="monospace",
            limit=50,
            angle=360 - 45,
        ).encode(
            x=x,
            y=alt.Y(f'perf_per_watt_{dtype}:Q'),
            text=alt.condition(nearest, 'name:N', alt.value(' ')),
            color='vendor',
        )
        
        return (points + txt).transform_calculate(
            calculate=f'datum.{measure} / datum.power_TDP',
            as_=f'perf_per_watt_{dtype}'
        )

    def floats():
        fp64 = perf_evol('performance_fp64')
        fp32 = perf_evol('performance_fp32')
        fp16 = perf_evol('performance_fp16')

        fp64_w = perf_per_watt('performance_fp64')
        fp32_w = perf_per_watt('performance_fp32')
        fp16_w = perf_per_watt('performance_fp16')


        chart = (fp64 | fp32 | fp16) & (fp64_w | fp32_w | fp16_w) 

        chart.show()

    def ints():
        int8 = perf_evol('performance_int8')
        int8_w = perf_per_watt('performance_int8')

        chart = int8 | int8_w
        chart.show()

    # ints()
    floats()


graphs()