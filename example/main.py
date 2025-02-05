from fastlyzer import Fastlyzer
import polars as pl

def calculation(a: int, b: int, c: int, d: float) -> dict:
    return { "result": a * 100 + b * 10 + c + d }

def main():
    fastlyzer = Fastlyzer(
        f=calculation,
        cache_file_name="fastlyze_cache.csv",
        input_schema=[
            ("a", pl.UInt8), 
            ("b", pl.UInt8), 
            ("c", pl.UInt8), 
            ("d", pl.Float64)
        ],
        output_schema=[
            ("result", pl.Float64)
        ]
    )

    fastlyzer.run({
        "a": [1, 2, 3],
        "b": [4, 5],
        "c": [6, 7],
        "d": 0.9,
    })

    result = fastlyzer.cache_table
    print(result)

if __name__ == "__main__":
    main()
