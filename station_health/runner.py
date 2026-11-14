"""运行领域摘要或启动本地供数服务。"""

import argparse
import json
from pathlib import Path
from http.server import ThreadingHTTPServer

from .analysis import summarize
from .datasource import SourceClient
from .reports import package_report, save_report
from .service import handler_for, load_repository, serve
from .valuation import assess


def main():
    parser = argparse.ArgumentParser(description="领域资料摘要与检修组合信息价值评估")
    parser.add_argument("--dataset", default="fixtures/dataset.json")
    parser.add_argument("--source-url")
    parser.add_argument("--output", default="/tmp/domain-summary.json")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--valuation", action="store_true",
                        help="输出基于历史观测的检修组合信息价值评估，而非现行资料摘要")
    parser.add_argument("--plan-id", help="多预算方案时指定评估的 plan_id")
    arguments = parser.parse_args()
    schema_path = Path(__file__).with_name("schema.json")
    if arguments.serve:
        server = ThreadingHTTPServer(("127.0.0.1", arguments.port), handler_for(load_repository(arguments.dataset, schema_path)))
        try:
            print(json.dumps({"listening": server.server_address}, ensure_ascii=False), flush=True)
            server.serve_forever()
        finally:
            server.server_close()
        return
    if arguments.source_url:
        snapshot = SourceClient(arguments.source_url).read()
    else:
        with serve(load_repository(arguments.dataset, schema_path)) as base_url:
            snapshot = SourceClient(base_url).read()
    if arguments.valuation:
        product = {"product": "repair_information_value",
                   "valuation": assess(snapshot.tables, arguments.plan_id)}
    else:
        product = {"product": "domain_summary", "summary": summarize(snapshot.tables)}
    report = package_report(snapshot, product)
    save_report(arguments.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
