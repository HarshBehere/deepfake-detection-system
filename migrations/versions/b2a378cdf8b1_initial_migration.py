"""Initial migration

Revision ID: b2a378cdf8b1
Revises: 
Create Date: 2025-04-27 13:25:46.977879

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy import JSON
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = 'b2a378cdf8b1'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns('user')]

    # Only add the column if it doesn't exist
    if 'upload_history' not in columns:
        op.add_column('user', sa.Column('upload_history', sa.JSON(), nullable=True))
