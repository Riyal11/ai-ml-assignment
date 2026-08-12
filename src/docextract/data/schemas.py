"""Pydantic v2 models for invoice extraction."""

import re
from datetime import date
from decimal import Decimal
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationInfo,
    field_validator,
)
from pydantic_core import PydanticCustomError

INVOICE_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")


def _validate_date_string(value: str) -> str:
    """Validate that ``value`` is a real calendar date in ISO format."""
    if not INVOICE_DATE_PATTERN.fullmatch(value):
        raise PydanticCustomError(
            "invalid_date_format",
            "invoice_date must match YYYY-MM-DD, got {value}",
            {"value": value},
        )
    date.fromisoformat(value)  # rejects impossible dates like 2024-02-31
    return value


def _strip(value: str) -> str:
    """Strip surrounding whitespace from a string field."""
    return value.strip()


DateStr = Annotated[
    str,
    AfterValidator(_validate_date_string),
    PlainSerializer(lambda v: v, return_type=str),
]

Money = Annotated[
    Decimal,
    PlainSerializer(lambda v: str(v), return_type=str, when_used="json"),
]

StrippedStr = Annotated[str, AfterValidator(_strip)]


class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: StrippedStr = Field(min_length=1)
    quantity: Decimal = Field(ge=0)
    unit_price: Decimal = Field(ge=0)

    model_config = ConfigDict(strict=True, extra="forbid")

    @field_validator("quantity", "unit_price", mode="before")
    @classmethod
    def _coerce_decimal(cls, value: object) -> object:
        # Strict Decimal rejects int/float; coerce numeric JSON input only.
        if isinstance(value, int | float):
            return Decimal(str(value))
        return value

    @field_validator("description", mode="after")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("description must not be blank after stripping")
        return value


class Invoice(BaseModel):
    """Structured metadata extracted from an invoice document."""

    invoice_number: StrippedStr = Field(min_length=1)
    vendor_name: StrippedStr = Field(min_length=1)
    invoice_date: DateStr
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Money = Field(ge=0)
    tax_amount: Money = Field(ge=0)
    total_amount: Money = Field(ge=0)
    currency: StrippedStr

    model_config = ConfigDict(strict=True, extra="forbid")

    @field_validator("subtotal", "tax_amount", "total_amount", mode="before")
    @classmethod
    def _coerce_money(cls, value: object) -> object:
        # Strict Decimal rejects int/float; coerce numeric JSON input only.
        if isinstance(value, int | float):
            return Decimal(str(value))
        return value

    @field_validator("invoice_date", mode="before")
    @classmethod
    def _validate_invoice_date(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip()
        if not INVOICE_DATE_PATTERN.fullmatch(value):
            raise ValueError(
                f"invoice_date must match YYYY-MM-DD, got {value!r} (field {info.field_name})"
            )
        date.fromisoformat(value)  # rejects impossible dates like 2024-02-31
        return value

    @field_validator("currency", mode="before")
    @classmethod
    def _validate_currency(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip()
        if not CURRENCY_PATTERN.fullmatch(value):
            raise ValueError(
                f"currency must be 3 uppercase ASCII letters, got {value!r} "
                f"(field {info.field_name})"
            )
        return value
