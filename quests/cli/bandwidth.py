import click

from quests.entropy import get_bandwidth

from .log import logger


@click.command("bandwidth")
@click.argument("atomic_volume", type=float, required=1)
@click.option(
    "-c",
    "--cutoff",
    is_flag=True,
    default=False,
    help="If True, uses the cutoff function instead of the Gaussian fit",
)
def bandwidth(atomic_volume, cutoff):
    method = "cutoff" if cutoff else "gaussian"
    bw = get_bandwidth(atomic_volume, method)
    logger(f"V = {atomic_volume:.2f} Å^3/atom -> h = {bw:.5f}")
