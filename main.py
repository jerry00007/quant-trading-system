"""A股量化交易系统 CLI"""
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    """A股量化交易系统"""
    pass


@cli.command()
def init():
    """初始化数据库"""
    from scripts.init_db import init_database

    console.print("[bold green]正在初始化数据库...[/bold green]")
    init_database()
    console.print("[bold green]✓ 数据库初始化完成[/bold green]")


@cli.command()
@click.argument("symbol", type=str)
@click.option("--days", default=365, help="拉取天数")
@click.option("--adjust", default="qfq", type=click.Choice(["qfq", "hfq", ""]), help="复权方式")
def daily(symbol, days, adjust):
    """拉取日线数据

    SYMBOL: 股票代码，如 600000
    """
    from data.downloader import AKShareDownloader
    from datetime import datetime, timedelta

    console.print(f"[bold]正在拉取 {symbol} 日线数据...[/bold]")

    downloader = AKShareDownloader()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    try:
        df = downloader.get_daily_quotes(
            symbol=symbol,
            start_date=start_date_str,
            end_date=end_date_str,
            adjust=adjust if adjust else None
        )

        if df.empty:
            console.print("[yellow]未获取到数据[/yellow]")
            return

        console.print(f"[green]✓ 获取成功，共 {len(df)} 条数据[/green]")
        console.print(f"\n[bold]数据列:[/bold] {', '.join(df.columns.tolist())}")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("日期")
        table.add_column("开盘")
        table.add_column("最高")
        table.add_column("最低")
        table.add_column("收盘")
        table.add_column("涨跌幅")

        for idx in range(min(10, len(df))):
            row = df.iloc[idx]
            table.add_row(
                str(row['日期']),
                f"{row['开盘']:.2f}",
                f"{row['最高']:.2f}",
                f"{row['最低']:.2f}",
                f"{row['收盘']:.2f}",
                f"{row['涨跌幅']:.2f}%"
            )

        console.print("\n[bold]最新 10 条数据:[/bold]")
        console.print(table)

    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        raise


@cli.command()
@click.argument("symbol", type=str)
@click.option("--start", help="开始日期，格式: YYYYMMDD")
@click.option("--end", help="结束日期，格式: YYYYMMDD")
def query(symbol, start, end):
    """查询本地数据

    SYMBOL: 股票代码
    """
    from data.storage import DataStorage

    console.print(f"[bold]查询 {symbol} 数据...[/bold]")

    storage = DataStorage()

    df = storage.get_daily_quotes(
        ts_code=symbol,
        start_date=start,
        end_date=end
    )

    if df.empty:
        console.print("[yellow]无数据[/yellow]")
        return

    console.print(f"[green]✓ 查询到 {len(df)} 条数据[/green]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("日期")
    table.add_column("开盘")
    table.add_column("最高")
    table.add_column("最低")
    table.add_column("收盘")
    table.add_column("涨跌幅")

    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        table.add_row(
            str(row['trade_date']),
            f"{row['open']:.2f}",
            f"{row['high']:.2f}",
            f"{row['low']:.2f}",
            f"{row['close']:.2f}",
            f"{row['pct_chg']:.2f}%"
        )

    console.print("\n[bold]最新 10 条数据:[/bold]")
    console.print(table)


@cli.command()
def list():
    """列出所有股票"""
    from data.downloader import AKShareDownloader

    console.print("[bold]正在获取股票列表...[/bold]")

    downloader = AKShareDownloader()

    try:
        df = downloader.get_stock_list()
        console.print(f"[green]✓ 获取成功，共 {len(df)} 只股票[/green]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("代码")
        table.add_column("名称")

        for idx in range(min(20, len(df))):
            row = df.iloc[idx]
            table.add_row(row['code'], row['name'])

        console.print("\n[bold]前 20 只股票:[/bold]")
        console.print(table)

    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        raise


if __name__ == "__main__":
    cli()
