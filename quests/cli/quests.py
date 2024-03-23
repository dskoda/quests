import click

from quests.cli.approx_dH import approx_dH
from quests.cli.bandwidth import bandwidth
from quests.cli.compute_dH import dH
from quests.cli.entropy import entropy
from quests.cli.entropy_sampler import entropy_sampler
from quests.cli.make_descriptors import make_descriptors


class QuestsGroup(click.Group):
    pass


@click.command(cls=QuestsGroup)
def quests():
    """Command line interface for quests"""


quests.add_command(entropy)
quests.add_command(entropy_sampler)
quests.add_command(make_descriptors)
quests.add_command(dH)
quests.add_command(approx_dH)
quests.add_command(bandwidth)


if __name__ == "__main__":
    quests()
