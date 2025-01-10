import subprocess
from tqdm import tqdm # type: ignore
from threadpoolctl import threadpool_limits # type: ignore

with threadpool_limits(limits=8):
    tw_res = []
    # 打印结果（可选）
    for k in tqdm(range(5)):
        res = []
        for i in tqdm(range(20)):
            result = subprocess.run(['python3', '../start_game.py'], capture_output=True, text=True)

            stdout = result.stdout
            res.append(stdout)
            
        res = [line.strip() for line in res]
        count = sum(1 for num in res if float(num) > 1000)
        print(count)
        tw_res.append(count)

    print(tw_res)