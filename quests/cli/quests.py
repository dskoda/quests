import click

from quests.cli.approx_dH import approx_dH
from quests.cli.bandwidth import bandwidth
from quests.cli.compute_dH import dH
from quests.cli.entropy import entropy
from quests.cli.entropy_sampler import entropy_sampler
from quests.cli.make_descriptors import make_descriptors
from quests.cli.overlap import overlap
from quests.cli.learning_curve import learning_curve
from quests.cli.mcmc import mcmc
from quests.cli.active_learning import active_learning
from quests.cli.compress import compress


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
quests.add_command(overlap)
quests.add_command(learning_curve)
quests.add_command(mcmc)
quests.add_command(active_learning)
quests.add_command(compress)


if __name__ == "__main__":
    quests()
