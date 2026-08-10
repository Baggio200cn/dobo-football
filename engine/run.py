"""一键入口：更新数据 → 回测评估 → 输出下一轮预测台账。
用法：
  python run.py            # 用已缓存数据跑全流程
  python run.py --refresh  # 强制重新下载数据
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import data, evaluate, predict

def main():
    refresh = "--refresh" in sys.argv
    print("=" * 48); print("① 更新数据"); print("=" * 48)
    data.update(refresh=refresh)
    m = data.load_matches()
    print(f"比赛样本 {len(m)} 场 · {m['Date'].min().date()} → {m['Date'].max().date()}")

    print("\n" + "=" * 48); print("② 回测评估（Brier / LogLoss vs 市场）"); print("=" * 48)
    evaluate.run()

    print("\n" + "=" * 48); print("③ 下一轮预测（写入台账）"); print("=" * 48)
    predict.run()

if __name__ == "__main__":
    main()
