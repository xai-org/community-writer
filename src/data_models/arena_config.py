import hashlib
import os
import tomllib
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, Literal
import re

from note_writer.note_length import LENGTH_RESTRICTION_PLACEHOLDER, NOTE_CHAR_LIMIT

from data_models.writer_data_models import MediaType, PostWithContext


ALLOCATION_START = 0
ALLOCATION_END = 1000
DEFAULT_ALLOWED_MEDIA_TYPES: list[MediaType] = ["photo", "video", "animated_gif"]
MultiNotePolicy = Literal["prioritize", "once_per_writer", "allow_revisions"]


SUBMISSION_PURPOSE = "submission"
FEED_PURPOSE = "feed"


def post_digest(post_id: int, purpose: str, modulus: int) -> int:
    salt = os.environ.get("POST_HASH_SALT")
    assert salt, "POST_HASH_SALT is not set; add it to .env"
    h = hashlib.sha256(f"{salt}:{purpose}:{post_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") % modulus


def _is_host_only(model_uri: str) -> bool:
    if not model_uri:
        return False
    if "://" in model_uri or "/" in model_uri or "?" in model_uri or "#" in model_uri:
        return False
    return re.fullmatch(r"[a-zA-Z0-9.-]+(:\d+)?", model_uri) is not None


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AliasedModelConfig(StrictConfigModel):
    model_alias: str = Field(
        pattern=r"^[a-z0-9_]+$",
        description="Alias naming the model for this component; resolved to the real model name "
        "via the MODEL_<ALIAS> environment variable after config load",
    )
    model_name: str = Field(
        default="",
        description="Resolved model name; populated from the environment, never set in the config file",
    )

    @model_validator(mode="before")
    @classmethod
    def _forbid_model_name_in_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and "model_name" in data:
            raise ValueError(
                "model_name cannot be set in the config file; set model_alias instead"
            )
        return data


class FeedDefinition(StrictConfigModel):
    name: str = Field(
        pattern=r"^[a-z0-9_]+$",
        description="Unique name for this feed",
    )
    feed_size: str = Field(
        description="The value to pass to the API's post_selection parameter (e.g. 'small' or 'large'). "
        "Feeds sharing the same feed_size are fetched once from the API and split client-side by language.",
    )
    included_languages: list[str] | None = Field(
        default=None,
        description="If set, only include posts whose language is in this list (e.g. ['en']). "
        "Mutually exclusive with excluded_languages.",
    )
    excluded_languages: list[str] = Field(
        default_factory=list,
        description="Language codes to filter out from fetched posts (e.g. ['en'] to exclude English). "
        "Mutually exclusive with included_languages.",
    )
    parent_feed: str | None = Field(
        default=None,
        description="Name of the parent feed, or None for root feeds",
    )
    enabled_ranges: list[tuple[int, int]] = Field(
        default=[],
        description="Hash ranges of posts to process from this feed. Posts outside these ranges "
        "are discarded at fetch time. Empty list (default) means the entire feed is processed. "
        "Uses the salted feed digest with modulus ALLOCATION_END (1000).",
    )
    co_threshold: float = Field(
        default=0.5,
        description="The minimum CO score required to submit a note for this feed",
    )
    max_post_age_seconds: float = Field(
        default=172800,
        description="Maximum age in seconds for posts to be eligible for this feed",
    )
    rl_rejector: str = Field(
        description="RLRejector to apply for this feed",
        default="",
    )
    rejector_pass_threshold: float = Field(
        default=0,
        description="The threshold for the rejector to pass a note for this feed",
    )
    screenshot_rejector: str = Field(
        description="ScreenshotRejector to apply for this feed",
        default="",
    )
    screenshot_rejector_pass_threshold: float = Field(
        default=0,
        description="The threshold for the screenshot rejector to pass a note for this feed",
    )
    recent_context_rejector: str = Field(
        description="RecentContextRejector to apply for this feed",
        default="",
    )
    recent_context_rejector_pass_threshold: float = Field(
        default=0,
        description="The threshold for the recent context rejector to pass a note. "
        "LIKELY_HELPFUL = 1, everything else = 0; higher = safer (same convention as other rejectors).",
    )
    submission_ranges: list[tuple[int, int]] = Field(
        default=[],
        description="List of (start, end) ranges of the hash space that are enabled for submission. "
        "An empty list means this feed is in dry-run mode (notes are written but not submitted).",
    )
    notable_post_range: tuple[float, float] | None = Field(
        default=None,
        description="Notable-post score range [floor, ceiling) for routing. "
        "Posts with notable_post_prediction in [floor, ceiling) are routed to this feed. "
        "None means accept all scores.",
    )
    notable_post_threshold: float | None = Field(
        default=None,
        description="Minimum notable-post prediction required to proceed with writing. "
        "Posts below this threshold are rejected pre-writing. "
        "Must be None when notable_post_range lower bound > 0; "
        "must be set when notable_post_range is None or starts at 0.",
    )

    @model_validator(mode="after")
    def validate_notable_post_range(self) -> "FeedDefinition":
        if self.notable_post_range is not None:
            floor, ceiling = self.notable_post_range
            if not (0 <= floor < ceiling <= 1):
                raise ValueError(
                    f"FeedDefinition '{self.name}' notable_post_range ({floor}, {ceiling}) "
                    f"must satisfy 0 <= floor < ceiling <= 1"
                )
        if self.notable_post_threshold is not None and not (
            0 <= self.notable_post_threshold <= 1
        ):
            raise ValueError(
                f"FeedDefinition '{self.name}' notable_post_threshold ({self.notable_post_threshold}) "
                f"must be in [0, 1]"
            )

        range_starts_above_zero = (
            self.notable_post_range is not None and self.notable_post_range[0] > 0
        )
        if range_starts_above_zero and self.notable_post_threshold is not None:
            raise ValueError(
                f"FeedDefinition '{self.name}': notable_post_threshold must be None "
                f"when notable_post_range lower bound is > 0 "
                f"(notable_post_range={self.notable_post_range}, "
                f"notable_post_threshold={self.notable_post_threshold})"
            )
        if not range_starts_above_zero and self.notable_post_threshold is None:
            raise ValueError(
                f"FeedDefinition '{self.name}': notable_post_threshold must be set "
                f"when notable_post_range is None or starts at 0"
            )
        return self

    @model_validator(mode="after")
    def validate_enabled_ranges(self) -> "FeedDefinition":
        for i, (start, end) in enumerate(self.enabled_ranges):
            if end <= start:
                raise ValueError(
                    f"FeedDefinition '{self.name}' enabled_ranges[{i}] has end ({end}) <= start ({start})"
                )
            if start < ALLOCATION_START:
                raise ValueError(
                    f"FeedDefinition '{self.name}' enabled_ranges[{i}] start ({start}) < ALLOCATION_START ({ALLOCATION_START})"
                )
            if end > ALLOCATION_END:
                raise ValueError(
                    f"FeedDefinition '{self.name}' enabled_ranges[{i}] end ({end}) > ALLOCATION_END ({ALLOCATION_END})"
                )

        for i in range(len(self.enabled_ranges) - 1):
            if self.enabled_ranges[i][1] >= self.enabled_ranges[i + 1][0]:
                raise ValueError(
                    f"FeedDefinition '{self.name}' enabled_ranges are not in sorted order: "
                    f"range[{i}] ends at {self.enabled_ranges[i][1]} >= range[{i + 1}] starts at {self.enabled_ranges[i + 1][0]}"
                )

        return self

    @model_validator(mode="after")
    def validate_language_filters(self) -> "FeedDefinition":
        if self.included_languages and self.excluded_languages:
            raise ValueError(
                f"FeedDefinition '{self.name}' has both included_languages and excluded_languages set. "
                f"These are mutually exclusive."
            )
        return self

    @model_validator(mode="after")
    def validate_rejector_threshold(self) -> "FeedDefinition":
        if self.rl_rejector and self.rejector_pass_threshold == 0:
            raise ValueError(
                f"rejector_pass_threshold must be set (non-zero) when rl_rejector is configured for feed '{self.name}'"
            )
        return self

    @model_validator(mode="after")
    def validate_screenshot_rejector_threshold(self) -> "FeedDefinition":
        if self.screenshot_rejector and self.screenshot_rejector_pass_threshold == 0:
            raise ValueError(
                f"screenshot_rejector_pass_threshold must be set (non-zero) when screenshot_rejector "
                f"is configured for feed '{self.name}'"
            )
        return self

    @model_validator(mode="after")
    def validate_recent_context_rejector_threshold(self) -> "FeedDefinition":
        if (
            self.recent_context_rejector
            and self.recent_context_rejector_pass_threshold == 0
        ):
            raise ValueError(
                f"recent_context_rejector_pass_threshold must be set (non-zero) when recent_context_rejector "
                f"is configured for feed '{self.name}'"
            )
        return self

    @model_validator(mode="after")
    def validate_submission_ranges(self) -> "FeedDefinition":
        for i, (start, end) in enumerate(self.submission_ranges):
            if end <= start:
                raise ValueError(
                    f"FeedDefinition '{self.name}' submission_ranges[{i}] has end ({end}) <= start ({start})"
                )
            if start < ALLOCATION_START:
                raise ValueError(
                    f"FeedDefinition '{self.name}' submission_ranges[{i}] start ({start}) < ALLOCATION_START ({ALLOCATION_START})"
                )
            if end > ALLOCATION_END:
                raise ValueError(
                    f"FeedDefinition '{self.name}' submission_ranges[{i}] end ({end}) > ALLOCATION_END ({ALLOCATION_END})"
                )

        for i in range(len(self.submission_ranges) - 1):
            if self.submission_ranges[i][1] >= self.submission_ranges[i + 1][0]:
                raise ValueError(
                    f"FeedDefinition '{self.name}' submission_ranges are not in sorted order: "
                    f"range[{i}] ends at {self.submission_ranges[i][1]} >= range[{i + 1}] starts at {self.submission_ranges[i + 1][0]}"
                )

        return self


class RevisionFeedDefinition(StrictConfigModel):
    name: str = Field(
        pattern=r"^[a-z0-9_]+$",
        description="Unique name for this revision feed",
    )
    latency_seconds: int = Field(
        gt=0,
        description="Delay in seconds after initial publication before revision work becomes eligible",
    )
    revision_rejector: str = Field(
        description="RevisionRejector to apply for this revision feed",
    )
    revision_rejector_pass_threshold: float = Field(
        gt=0,
        description="The threshold for the revision rejector to pass a note. "
        "Applied to the scaled score: probability * (1 - sim_no_urls).",
    )


class RetryFeedDefinition(StrictConfigModel):
    name: str = Field(
        pattern=r"^[a-z0-9_]+$",
        description="Unique name for this retry feed",
    )
    latency_seconds: int = Field(
        gt=0,
        description="Delay in seconds after API feed enqueue before retry becomes eligible",
    )
    max_post_age_seconds: float = Field(
        gt=0,
        description="Maximum post age in seconds (post creation to work_started_at). "
        "Posts older than this are not eligible for retry.",
    )


class FeedDefaults(StrictConfigModel):
    co_threshold: float | None = None
    max_post_age_seconds: float | None = None
    rl_rejector: str | None = None
    rejector_pass_threshold: float | None = None
    screenshot_rejector: str | None = None
    screenshot_rejector_pass_threshold: float | None = None
    recent_context_rejector: str | None = None
    recent_context_rejector_pass_threshold: float | None = None
    submission_ranges: list[tuple[int, int]] | None = None
    notable_post_threshold: float | None = None


class GrokWriter(AliasedModelConfig):
    writer_name: str = Field(
        pattern=r"^[a-z0-9_]+$", description="The name of the writer"
    )
    model_uri: str = Field(
        default="api.x.ai", description="The host or base URI of the model API"
    )
    writer_prompt: str = Field(
        description="The prompt template to use for note writing (must support '{post_link}' placeholder)",
        examples="Write a note for the following post: {post_link}",
    )
    length_restriction: str = Field(
        default="",
        description="Text substituted into the writer_prompt's '{length_restriction}' slot when every "
        "draft in the first round exceeded the note length limit, prompting a second round. The slot is "
        "left empty in the first round, so the restriction only appears once it is needed. Configuring "
        "one of the two without the other is a validation error; leaving both out disables the retry.",
    )
    allowed_media_types: list[MediaType] = Field(
        description="List of media types to allow posts",
        default_factory=lambda: list(DEFAULT_ALLOWED_MEDIA_TYPES),
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature to use for the LLM",
    )
    num_drafts: int = Field(
        default=3, description="The number of draft notes to generate"
    )
    timeout: float = Field(
        default=600,
        description="The timeout in seconds for the LLM",
    )
    max_retries_llm: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls",
    )
    retry_base_delay: float = Field(
        default=0.0,
        description="Base delay in seconds added to every exponential backoff sleep between retries",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, this writer generates drafts but skips submission. Results are "
        "recorded in the feed parquet with a dry_run column indicating what would have happened. "
        "The global --dry-run CLI flag sets effective dry_run to True for all writers.",
    )
    api_feeds: list[str] = Field(
        default=[],
        description="List of API feed names this writer subscribes to",
    )
    revision_feeds: list[str] = Field(
        default=[],
        description="List of revision feed names this writer subscribes to",
    )
    retry_feeds: list[str] = Field(
        default=[],
        description="List of retry feed names this writer subscribes to",
    )
    enabled_ranges: list[tuple[int, int]] = Field(
        default=[],
        description="Hash ranges of posts this writer should process. Posts outside these ranges "
        "are skipped entirely (no drafting, no submission). Empty list (default) means the "
        "writer processes all posts from its subscribed feeds. "
        "Uses the salted submission digest with modulus ALLOCATION_END (1000), same as multi_note_policy.",
    )

    @model_validator(mode="after")
    def validate_enabled_ranges(self) -> "GrokWriter":
        for i, (start, end) in enumerate(self.enabled_ranges):
            if end <= start:
                raise ValueError(
                    f"Writer '{self.writer_name}' enabled_ranges[{i}] has end ({end}) <= start ({start})"
                )
            if start < ALLOCATION_START:
                raise ValueError(
                    f"Writer '{self.writer_name}' enabled_ranges[{i}] start ({start}) < ALLOCATION_START ({ALLOCATION_START})"
                )
            if end > ALLOCATION_END:
                raise ValueError(
                    f"Writer '{self.writer_name}' enabled_ranges[{i}] end ({end}) > ALLOCATION_END ({ALLOCATION_END})"
                )
        for i in range(len(self.enabled_ranges) - 1):
            if self.enabled_ranges[i][1] > self.enabled_ranges[i + 1][0]:
                raise ValueError(
                    f"Writer '{self.writer_name}' enabled_ranges are not sorted/non-overlapping: "
                    f"range[{i}] ends at {self.enabled_ranges[i][1]} > range[{i + 1}] starts at {self.enabled_ranges[i + 1][0]}"
                )
        return self

    @model_validator(mode="after")
    def validate_feed_subscriptions(self) -> "GrokWriter":
        if not self.dry_run and not self.api_feeds:
            raise ValueError(
                f"Writer '{self.writer_name}' has dry_run=False but no api_feeds subscribed"
            )
        for attr in ("api_feeds", "revision_feeds", "retry_feeds"):
            names = getattr(self, attr)
            if len(names) != len(set(names)):
                duplicates = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"Writer '{self.writer_name}' has duplicate names in {attr}: {duplicates}"
                )
        return self

    @model_validator(mode="after")
    def validate_length_restriction(self) -> "GrokWriter":
        has_slot = LENGTH_RESTRICTION_PLACEHOLDER in self.writer_prompt
        if has_slot and not self.length_restriction:
            raise ValueError(
                f"Writer '{self.writer_name}' has a {LENGTH_RESTRICTION_PLACEHOLDER} slot in its "
                "writer_prompt but no length_restriction text to put in it"
            )
        if self.length_restriction and not has_slot:
            raise ValueError(
                f"Writer '{self.writer_name}' sets length_restriction but its writer_prompt has no "
                f"{LENGTH_RESTRICTION_PLACEHOLDER} slot, so the text would never be used"
            )
        if (
            self.length_restriction
            and str(NOTE_CHAR_LIMIT) not in self.length_restriction
        ):
            raise ValueError(
                f"Writer '{self.writer_name}' length_restriction does not mention the enforced limit "
                f"of {NOTE_CHAR_LIMIT} characters"
            )
        return self

    @model_validator(mode="after")
    def validate_client_settings(self) -> "GrokWriter":
        if not _is_host_only(self.model_uri):
            raise ValueError(
                f"Writer '{self.writer_name}' has model_uri='{self.model_uri}', which is not a host-only value"
            )
        return self


class SubmissionConfig(StrictConfigModel):
    account_name: str = Field(
        description="Suffix used to define environment variables for the submission account",
    )
    daily_limit: int = Field(
        default=50,
        description="Maximum number of notes that can be submitted within a 24 hour period for this account.",
    )


class RLRejector(AliasedModelConfig):
    rejector_name: str = Field(description="The name of the rejector")
    model_uri: str = Field(
        default="api.x.ai", description="The host or base URI of the model API"
    )
    rejector_prompt: str = Field(
        description="The prompt template to use for note rejection (must support '{post_link}' and '{note_text}' placeholders)",
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature to use for the LLM",
    )
    num_samples: int = Field(
        description="The number of samples to take for the rejector",
    )
    timeout: float = Field(
        default=600,
        description="The timeout in seconds for the LLM",
    )
    max_retries_llm: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls",
    )
    retry_base_delay: float = Field(
        default=0.0,
        description="Base delay in seconds added to every exponential backoff sleep between retries",
    )

    @model_validator(mode="after")
    def validate_client_settings(self) -> "RLRejector":
        if not _is_host_only(self.model_uri):
            raise ValueError(
                f"Rejector '{self.rejector_name}' has model_uri='{self.model_uri}', which is not a host-only value"
            )
        return self


class ScreenshotRejector(AliasedModelConfig):
    rejector_name: str = Field(description="The name of the screenshot rejector")
    model_uri: str = Field(
        default="api.x.ai", description="The host or base URI of the model API"
    )
    rejector_prompt: str = Field(
        description="The prompt template to use for screenshot-based note rejection "
        "(must support '{post_link}' and '{note_text}' placeholders)",
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature to use for the LLM",
    )
    min_samples: int = Field(
        description="Number of samples to take in the first round. If any first-round sample "
        "scores below the pass threshold, additional samples are taken up to max_samples.",
    )
    max_samples: int = Field(
        description="Maximum total number of samples across both rounds. The second round is only "
        "issued when a first-round sample scores below the pass threshold.",
    )
    timeout: float = Field(
        default=600,
        description="The timeout in seconds for the LLM",
    )
    max_retries_llm: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls",
    )
    retry_base_delay: float = Field(
        default=0.0,
        description="Base delay in seconds added to every exponential backoff sleep between retries",
    )

    page_timeout: int = Field(
        default=30000,
        description="Page load timeout in milliseconds for Playwright",
    )
    post_load_delay: int = Field(
        default=2000,
        description="Delay in ms after network idle for the third screenshot",
    )
    max_retries: int = Field(
        default=3,
        description="Number of retries for failed screenshot captures",
    )
    max_screenshot_height: int = Field(
        default=5000,
        description="Maximum height in pixels for captured screenshots",
    )

    @model_validator(mode="after")
    def validate_client_settings(self) -> "ScreenshotRejector":
        if not _is_host_only(self.model_uri):
            raise ValueError(
                f"ScreenshotRejector '{self.rejector_name}' has model_uri='{self.model_uri}' "
                f"which is not a host-only value"
            )
        if self.min_samples < 1:
            raise ValueError(
                f"ScreenshotRejector '{self.rejector_name}' has min_samples={self.min_samples}; "
                f"must be >= 1"
            )
        if self.max_samples < self.min_samples:
            raise ValueError(
                f"ScreenshotRejector '{self.rejector_name}' has max_samples={self.max_samples} "
                f"which is less than min_samples={self.min_samples}"
            )
        return self


class RecentContextRejector(AliasedModelConfig):
    rejector_name: str = Field(description="The name of the recent context rejector")
    model_uri: str = Field(
        default="api.x.ai", description="The host or base URI of the model API"
    )
    rejector_prompt: str = Field(
        description="System prompt for the classifier. The user message with prior "
        "examples and candidate note is built dynamically at query time.",
    )
    temperature: float = Field(
        default=1.0,
        description="The temperature to use for the LLM",
    )
    num_samples: int = Field(
        default=5,
        description="The number of samples to take for the rejector",
    )
    timeout: float = Field(
        default=600,
        description="The timeout in seconds for the LLM",
    )
    max_retries_llm: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls",
    )
    retry_base_delay: float = Field(
        default=0.0,
        description="Base delay in seconds added to every exponential backoff sleep between retries",
    )
    crh_count: int = Field(
        default=100,
        description="Number of recent CRH examples to include as context",
    )
    crnh_count: int = Field(
        default=100,
        description="Number of recent CRNH examples to include as context",
    )

    @model_validator(mode="after")
    def validate_client_settings(self) -> "RecentContextRejector":
        if not _is_host_only(self.model_uri):
            raise ValueError(
                f"RecentContextRejector '{self.rejector_name}' has model_uri='{self.model_uri}' "
                f"which is not a host-only value"
            )
        return self


class RevisionRejector(AliasedModelConfig):
    rejector_name: str = Field(description="The name of the revision rejector")
    model_uri: str = Field(
        default="api.x.ai", description="The host or base URI of the model API"
    )
    rejector_prompt: str = Field(
        description="The prompt template for the revision rejector "
        "(must support '{post_link}', '{baseline_note}', '{candidate_note}' placeholders)",
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature to use for the LLM",
    )
    num_samples: int = Field(
        default=1,
        description="The number of samples to take for the rejector",
    )
    timeout: float = Field(
        default=600,
        description="The timeout in seconds for the LLM",
    )
    max_retries_llm: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls",
    )
    retry_base_delay: float = Field(
        default=0.0,
        description="Base delay in seconds added to every exponential backoff sleep between retries",
    )

    @model_validator(mode="after")
    def validate_client_settings(self) -> "RevisionRejector":
        if not _is_host_only(self.model_uri):
            raise ValueError(
                f"RevisionRejector '{self.rejector_name}' has model_uri='{self.model_uri}' "
                f"which is not a host-only value"
            )
        return self


class BaseDeletionPolicy(StrictConfigModel):
    policy_name: str = Field(
        pattern=r"^[a-z0-9_]+$",
        description="Unique name for this deletion policy.",
    )
    max_age_days: int = Field(
        default=3,
        description="Only consider notes created within the last N days for deletion (default: 3).",
    )
    max_deletion_fraction: float = Field(
        default=0.2,
        description="Safeguard: if more than this fraction of eligible-age notes would be deleted, "
        "log an error and skip all deletions for this policy (default: 0.2 = 20%).",
    )


class ModelDeletionPolicy(BaseDeletionPolicy):
    min_total_ratings: int = Field(
        description="Minimum total rating count required for this policy to apply.",
    )
    score_threshold: float = Field(
        description="Minimum model score (P(not CRH)) required to trigger deletion.",
    )

    @model_validator(mode="after")
    def validate_threshold(self) -> "ModelDeletionPolicy":
        if self.min_total_ratings < 1:
            raise ValueError(
                f"ModelDeletionPolicy '{self.policy_name}': "
                f"min_total_ratings must be >= 1, got {self.min_total_ratings}"
            )
        if not (0.0 < self.score_threshold <= 1.0):
            raise ValueError(
                f"ModelDeletionPolicy '{self.policy_name}': "
                f"score_threshold must be in (0, 1], got {self.score_threshold}"
            )
        return self


class ArenaConfig(StrictConfigModel):
    config_timestamp: int = Field(
        default=0,
        description="Timestamp (seconds since epoch) of when this config was created. Set from filename when loading.",
    )
    max_configured_writers_per_post: int = Field(
        default=1,
        description="Maximum number of non-dry-run writers that can be assigned to any point in the hash space. "
        "Validated at config time.",
    )
    max_published_notes_per_post: int = Field(
        default=1,
        description="Maximum total number of notes that can be submitted per post (including deleted notes). "
        "Checked at submission time.",
    )
    max_current_notes_per_post: int = Field(
        default=1,
        description="Maximum number of non-deleted notes that can exist on a post. "
        "Checked at submission time.",
    )
    api_feed_defaults: FeedDefaults | None = Field(
        default=None,
        description="Default values for FeedDefinition config fields. Any field in an api_feeds "
        "entry that is not explicitly set will be filled from these defaults.",
    )
    api_feeds: list[FeedDefinition] = Field(
        description="List of API feed definitions",
        min_length=1,
    )
    rl_rejectors: list[RLRejector] = Field(
        description="List of Grok rejectors to use",
        min_length=1,
    )
    screenshot_rejectors: list[ScreenshotRejector] = Field(
        description="List of screenshot rejectors to use",
        default_factory=list,
    )
    recent_context_rejectors: list[RecentContextRejector] = Field(
        description="List of recent context rejectors to use",
        default_factory=list,
    )
    revision_rejectors: list[RevisionRejector] = Field(
        description="List of revision rejectors to use",
        default_factory=list,
    )
    grok_writers: list[GrokWriter] = Field(
        description="List of Grok writers to use",
        min_length=1,
    )
    submission_configs: list[SubmissionConfig] = Field(
        description="List of submission configurations",
        min_length=1,
    )
    writer_priority: list[str] = Field(
        default=[],
        description="Ordered list of all grok_writer names defining priority. "
        "Must contain every writer exactly once, with all dry_run=False writers before all dry_run=True writers.",
    )
    multi_note_policy: list[tuple[int, int, MultiNotePolicy]] = Field(
        default_factory=lambda: [(ALLOCATION_START, ALLOCATION_END, "once_per_writer")],
        description="List of (start, end, policy) tuples defining the writer selection policy for each segment "
        "of the hash range. 'prioritize' uses the writer_priority list to select a single writer; "
        "'once_per_writer' allows each writer to submit at most once per post; "
        "'allow_revisions' allows additional notes after the first if the revision rejector passes.",
    )
    model_deletion_policies: list[ModelDeletionPolicy] = Field(
        default_factory=list,
        description="Model-based deletion policies. Each policy triggers deletion when the "
        "deletion model score exceeds a threshold at a given rating count.",
    )
    revision_feeds: list[RevisionFeedDefinition] = Field(
        default_factory=list,
        description="Revision feeds that re-process posts after a configured delay. "
        "Each entry defines a named feed with a latency in seconds.",
    )
    retry_feeds: list[RetryFeedDefinition] = Field(
        default_factory=list,
        description="Retry feeds that re-attempt note publication after a configured delay "
        "for posts that are still young enough. Each entry defines a named feed with a "
        "latency and a max post age.",
    )
    feed_size_priority: list[str] = Field(
        default_factory=list,
        description="Ordered list of feed sizes from highest to lowest priority. "
        "Used to determine which API feed's thresholds apply to timed-feed items. "
        "Must contain exactly the set of feed_size values from api_feeds.",
    )

    def get_multi_note_policy(self, post_id: int) -> MultiNotePolicy:
        digest = post_digest(post_id, SUBMISSION_PURPOSE, ALLOCATION_END)
        for start, end, policy in self.multi_note_policy:
            if digest >= start and digest < end:
                return policy
        raise ValueError(f"No multi_note_policy covers post digest {digest}")

    def get_api_feed_def(self, name: str) -> FeedDefinition:
        for f in self.api_feeds:
            if f.name == name:
                return f
        raise ValueError(f"Unknown API feed: '{name}'")

    @model_validator(mode="before")
    @classmethod
    def apply_api_feed_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        api_feed_defaults = data.get("api_feed_defaults")
        if not api_feed_defaults or not isinstance(api_feed_defaults, dict):
            return data
        for feed in data.get("api_feeds", []):
            if not isinstance(feed, dict):
                continue
            for key, value in api_feed_defaults.items():
                if key not in feed:
                    feed[key] = value
        return data

    @model_validator(mode="after")
    def validate_writer_priority(self) -> "ArenaConfig":
        writer_names = {w.writer_name for w in self.grok_writers}
        dry_run_names = {w.writer_name for w in self.grok_writers if w.dry_run}

        if len(self.writer_priority) != len(set(self.writer_priority)):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for name in self.writer_priority:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            raise ValueError(
                f"Duplicate entries in writer_priority: {sorted(duplicates)}"
            )

        priority_set = set(self.writer_priority)

        unknown_names = priority_set - writer_names
        if unknown_names:
            raise ValueError(
                f"writer_priority contains unknown writer names: {sorted(unknown_names)}. "
                f"Valid writers are: {sorted(writer_names)}"
            )

        missing = writer_names - priority_set
        if missing:
            raise ValueError(f"writer_priority is missing writers: {sorted(missing)}")

        seen_dry_run = False
        for name in self.writer_priority:
            is_dry_run = name in dry_run_names
            if seen_dry_run and not is_dry_run:
                raise ValueError(
                    f"writer_priority has non-dry-run writer '{name}' after a dry-run writer. "
                    f"All dry_run=False writers must appear before all dry_run=True writers."
                )
            if is_dry_run:
                seen_dry_run = True

        return self

    @model_validator(mode="after")
    def validate_multi_note_policy(self) -> "ArenaConfig":
        if not self.multi_note_policy:
            raise ValueError("multi_note_policy must not be empty")

        sorted_ranges = sorted(self.multi_note_policy, key=lambda r: r[0])

        for i, (start, end, _) in enumerate(sorted_ranges):
            if end <= start:
                raise ValueError(
                    f"multi_note_policy[{i}] has end ({end}) <= start ({start})"
                )
            if start < ALLOCATION_START:
                raise ValueError(
                    f"multi_note_policy[{i}] start ({start}) < ALLOCATION_START ({ALLOCATION_START})"
                )
            if end > ALLOCATION_END:
                raise ValueError(
                    f"multi_note_policy[{i}] end ({end}) > ALLOCATION_END ({ALLOCATION_END})"
                )

        if sorted_ranges[0][0] != ALLOCATION_START:
            raise ValueError(
                f"multi_note_policy must start at ALLOCATION_START ({ALLOCATION_START}), "
                f"but first range starts at {sorted_ranges[0][0]}"
            )

        if sorted_ranges[-1][1] != ALLOCATION_END:
            raise ValueError(
                f"multi_note_policy must end at ALLOCATION_END ({ALLOCATION_END}), "
                f"but last range ends at {sorted_ranges[-1][1]}"
            )

        for i in range(len(sorted_ranges) - 1):
            current_end = sorted_ranges[i][1]
            next_start = sorted_ranges[i + 1][0]
            if current_end > next_start:
                raise ValueError(
                    f"multi_note_policy overlap: range ending at {current_end} "
                    f"overlaps with range starting at {next_start}"
                )
            if current_end < next_start:
                raise ValueError(
                    f"multi_note_policy has a gap between {current_end} and {next_start}"
                )

        return self

    @model_validator(mode="after")
    def validate_unique_names(self) -> "ArenaConfig":
        all_names: list[str] = (
            [w.writer_name for w in self.grok_writers]
            + [r.rejector_name for r in self.rl_rejectors]
            + [s.rejector_name for s in self.screenshot_rejectors]
            + [r.rejector_name for r in self.recent_context_rejectors]
            + [r.rejector_name for r in self.revision_rejectors]
        )
        if len(all_names) != len(set(all_names)):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for name in all_names:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            raise ValueError(
                f"Duplicate names found across writers and rejectors: {sorted(duplicates)}"
            )
        return self

    @model_validator(mode="after")
    def validate_rl_rejector_references(self) -> "ArenaConfig":
        valid_rejector_names = {
            rejector.rejector_name for rejector in self.rl_rejectors
        }

        for feed_def in self.api_feeds:
            if (
                feed_def.rl_rejector
                and feed_def.rl_rejector not in valid_rejector_names
            ):
                raise ValueError(
                    f"Feed '{feed_def.name}' references unknown rl_rejector "
                    f"'{feed_def.rl_rejector}'. Valid rejectors are: {sorted(valid_rejector_names)}"
                )

        return self

    @model_validator(mode="after")
    def validate_screenshot_rejector_references(self) -> "ArenaConfig":
        valid_rejector_names = {
            rejector.rejector_name for rejector in self.screenshot_rejectors
        }

        for feed_def in self.api_feeds:
            if (
                feed_def.screenshot_rejector
                and feed_def.screenshot_rejector not in valid_rejector_names
            ):
                raise ValueError(
                    f"Feed '{feed_def.name}' references unknown screenshot_rejector "
                    f"'{feed_def.screenshot_rejector}'. Valid screenshot rejectors are: {sorted(valid_rejector_names)}"
                )

        return self

    @model_validator(mode="after")
    def validate_recent_context_rejector_references(self) -> "ArenaConfig":
        valid_rejector_names = {
            rejector.rejector_name for rejector in self.recent_context_rejectors
        }

        for feed_def in self.api_feeds:
            if (
                feed_def.recent_context_rejector
                and feed_def.recent_context_rejector not in valid_rejector_names
            ):
                raise ValueError(
                    f"Feed '{feed_def.name}' references unknown recent_context_rejector "
                    f"'{feed_def.recent_context_rejector}'. Valid recent context rejectors are: {sorted(valid_rejector_names)}"
                )

        return self

    @model_validator(mode="after")
    def validate_revision_rejector_references(self) -> "ArenaConfig":
        valid_rejector_names = {
            rejector.rejector_name for rejector in self.revision_rejectors
        }

        for rf in self.revision_feeds:
            if rf.revision_rejector not in valid_rejector_names:
                raise ValueError(
                    f"Revision feed '{rf.name}' references unknown revision_rejector "
                    f"'{rf.revision_rejector}'. Valid revision rejectors are: {sorted(valid_rejector_names)}"
                )

        return self

    @model_validator(mode="after")
    def validate_feed_names(self) -> "ArenaConfig":
        feed_names = [f.name for f in self.api_feeds]
        if len(feed_names) != len(set(feed_names)):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for name in feed_names:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            raise ValueError(f"Duplicate feed names: {sorted(duplicates)}")

        revision_feed_names = [rf.name for rf in self.revision_feeds]
        if len(revision_feed_names) != len(set(revision_feed_names)):
            seen2: set[str] = set()
            duplicates2: set[str] = set()
            for name in revision_feed_names:
                if name in seen2:
                    duplicates2.add(name)
                seen2.add(name)
            raise ValueError(f"Duplicate revision feed names: {sorted(duplicates2)}")

        retry_feed_names = [rf.name for rf in self.retry_feeds]
        if len(retry_feed_names) != len(set(retry_feed_names)):
            seen3: set[str] = set()
            duplicates3: set[str] = set()
            for name in retry_feed_names:
                if name in seen3:
                    duplicates3.add(name)
                seen3.add(name)
            raise ValueError(f"Duplicate retry feed names: {sorted(duplicates3)}")

        all_timed = set(revision_feed_names) | set(retry_feed_names)
        overlap = set(feed_names) & all_timed
        if overlap:
            raise ValueError(
                f"Timed feed names overlap with logical feed names: {sorted(overlap)}. "
                f"Timed feed names must be distinct from all logical feed names."
            )

        timed_overlap = set(revision_feed_names) & set(retry_feed_names)
        if timed_overlap:
            raise ValueError(
                f"Revision and retry feed names overlap: {sorted(timed_overlap)}. "
                f"All timed feed names must be unique."
            )

        return self

    @model_validator(mode="after")
    def validate_feed_hierarchy(self) -> "ArenaConfig":
        feed_names = {f.name for f in self.api_feeds}
        feed_by_name = {f.name: f for f in self.api_feeds}

        for feed in self.api_feeds:
            if feed.parent_feed is not None:
                if feed.parent_feed not in feed_names:
                    raise ValueError(
                        f"Feed '{feed.name}' references unknown parent_feed '{feed.parent_feed}'. "
                        f"Valid feeds are: {sorted(feed_names)}"
                    )
                parent = feed_by_name[feed.parent_feed]
                if parent.parent_feed is not None:
                    raise ValueError(
                        f"Feed '{feed.name}' sets parent_feed='{feed.parent_feed}', but "
                        f"'{feed.parent_feed}' is itself a child feed (parent_feed='{parent.parent_feed}'). "
                        f"parent_feed must reference a root feed (one with no parent)."
                    )

                boundary_points: set[int] = set()
                for start, end in feed.submission_ranges:
                    boundary_points.add(start)
                    boundary_points.add(end)
                for start, end in parent.submission_ranges:
                    boundary_points.add(start)
                    boundary_points.add(end)
                for bp in boundary_points:
                    in_child = any(
                        start <= bp < end for start, end in feed.submission_ranges
                    )
                    if not in_child:
                        continue
                    in_parent = any(
                        start <= bp < end for start, end in parent.submission_ranges
                    )
                    if not in_parent:
                        raise ValueError(
                            f"Feed '{feed.name}' submission_ranges must be a subset of "
                            f"its parent feed '{feed.parent_feed}'"
                        )
        return self

    @model_validator(mode="after")
    def validate_feed_references(self) -> "ArenaConfig":
        api_feed_names = {f.name for f in self.api_feeds}
        revision_feed_names = {rf.name for rf in self.revision_feeds}
        retry_feed_names = {rf.name for rf in self.retry_feeds}

        for writer in self.grok_writers:
            for name in writer.api_feeds:
                if name not in api_feed_names:
                    raise ValueError(
                        f"Writer '{writer.writer_name}' references unknown feed '{name}' "
                        f"in api_feeds. Valid API feeds are: {sorted(api_feed_names)}"
                    )
            for name in writer.revision_feeds:
                if name not in revision_feed_names:
                    raise ValueError(
                        f"Writer '{writer.writer_name}' references unknown feed '{name}' "
                        f"in revision_feeds. Valid revision feeds are: {sorted(revision_feed_names)}"
                    )
            for name in writer.retry_feeds:
                if name not in retry_feed_names:
                    raise ValueError(
                        f"Writer '{writer.writer_name}' references unknown feed '{name}' "
                        f"in retry_feeds. Valid retry feeds are: {sorted(retry_feed_names)}"
                    )

        return self

    @model_validator(mode="after")
    def validate_feeds_have_writers(self) -> "ArenaConfig":
        writer_feeds: set[str] = set()
        for writer in self.grok_writers:
            writer_feeds.update(writer.api_feeds)

        configured_feed_names = {f.name for f in self.api_feeds}
        feeds_without_writers = configured_feed_names - writer_feeds
        if feeds_without_writers:
            raise ValueError(
                f"The following feeds have no writers configured: {sorted(feeds_without_writers)}. "
                f"Either add writers for these feeds or remove them from the api_feeds list."
            )
        return self

    @model_validator(mode="after")
    def validate_writer_feed_hierarchy(self) -> "ArenaConfig":
        feed_by_name = {f.name: f for f in self.api_feeds}

        for writer in self.grok_writers:
            if not writer.api_feeds:
                continue

            writer_feed_names = set(writer.api_feeds)
            for feed_name in writer.api_feeds:
                feed_def = feed_by_name.get(feed_name)
                if feed_def is not None and feed_def.parent_feed is not None:
                    if feed_def.parent_feed not in writer_feed_names:
                        raise ValueError(
                            f"Writer '{writer.writer_name}' subscribes to feed '{feed_name}' "
                            f"but not its parent feed '{feed_def.parent_feed}'"
                        )
        return self

    @model_validator(mode="after")
    def validate_maximum_notes_per_post(self) -> "ArenaConfig":
        root_feeds = [f for f in self.api_feeds if f.parent_feed is None]

        for root_feed in root_feeds:
            if not root_feed.submission_ranges:
                continue

            active_writers = [
                w
                for w in self.grok_writers
                if not w.dry_run and root_feed.name in w.api_feeds
            ]

            boundary_points: set[int] = {ALLOCATION_START, ALLOCATION_END}
            for w in active_writers:
                for start, end in w.enabled_ranges:
                    boundary_points.add(start)
                    boundary_points.add(end)
            sorted_points = sorted(boundary_points)

            submission_set: set[int] = set()
            for start, end in root_feed.submission_ranges:
                for bp in sorted_points:
                    if start <= bp < end:
                        submission_set.add(bp)

            for i in range(len(sorted_points) - 1):
                test_point = sorted_points[i]
                num_writers = sum(
                    1
                    for w in active_writers
                    if not w.enabled_ranges
                    or any(start <= test_point < end for start, end in w.enabled_ranges)
                )
                if num_writers > self.max_configured_writers_per_post:
                    raise ValueError(
                        f"Root feed '{root_feed.name}': {num_writers} non-dry-run writers "
                        f"are active at hash point {test_point}, which exceeds "
                        f"max_configured_writers_per_post={self.max_configured_writers_per_post}"
                    )
                if test_point in submission_set and num_writers == 0:
                    raise ValueError(
                        f"Root feed '{root_feed.name}' has submission_ranges covering "
                        f"hash point {test_point} but no non-dry-run writers are active there"
                    )

        once_per_writer_segments = [
            (start, end)
            for start, end, policy in self.multi_note_policy
            if policy == "once_per_writer"
        ]
        if once_per_writer_segments:
            all_non_dry_run = [w for w in self.grok_writers if not w.dry_run]

            boundary_points_opw: set[int] = set()
            for seg_start, seg_end in once_per_writer_segments:
                boundary_points_opw.add(seg_start)
                boundary_points_opw.add(seg_end)
            for w in all_non_dry_run:
                for start, end in w.enabled_ranges:
                    boundary_points_opw.add(start)
                    boundary_points_opw.add(end)
            sorted_points_opw = sorted(boundary_points_opw)

            for i in range(len(sorted_points_opw) - 1):
                test_point = sorted_points_opw[i]

                in_opw = any(
                    seg_start <= test_point < seg_end
                    for seg_start, seg_end in once_per_writer_segments
                )
                if not in_opw:
                    continue
                num_writers = sum(
                    1
                    for w in all_non_dry_run
                    if not w.enabled_ranges
                    or any(start <= test_point < end for start, end in w.enabled_ranges)
                )
                if num_writers > self.max_configured_writers_per_post:
                    raise ValueError(
                        f"once_per_writer segment: {num_writers} non-dry-run writers "
                        f"are active at hash point {test_point} (feed-agnostic), "
                        f"which exceeds max_configured_writers_per_post="
                        f"{self.max_configured_writers_per_post}"
                    )

        return self

    @model_validator(mode="after")
    def validate_deletion_policies(self) -> "ArenaConfig":
        all_names = [p.policy_name for p in self.model_deletion_policies]
        if len(all_names) != len(set(all_names)):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for name in all_names:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            raise ValueError(f"Duplicate deletion policy names: {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def validate_revision_feeds_have_revision_rejector(self) -> "ArenaConfig":
        for rf in self.revision_feeds:
            if not rf.revision_rejector:
                raise ValueError(
                    f"Revision feed '{rf.name}' has an empty revision_rejector. "
                    f"Revision feeds require a revision_rejector."
                )
        return self

    @model_validator(mode="after")
    def validate_feed_size_priority(self) -> "ArenaConfig":
        if not self.feed_size_priority:
            return self
        config_sizes = {f.feed_size for f in self.api_feeds}
        priority_sizes = set(self.feed_size_priority)
        if len(self.feed_size_priority) != len(priority_sizes):
            duplicates = sorted(
                s
                for s in self.feed_size_priority
                if self.feed_size_priority.count(s) > 1
            )
            raise ValueError(f"Duplicate entries in feed_size_priority: {duplicates}")
        unknown = priority_sizes - config_sizes
        if unknown:
            raise ValueError(
                f"feed_size_priority contains unknown feed sizes: {sorted(unknown)}. "
                f"Valid sizes are: {sorted(config_sizes)}"
            )
        missing = config_sizes - priority_sizes
        if missing:
            raise ValueError(
                f"feed_size_priority is missing feed sizes: {sorted(missing)}. "
                f"All feed sizes from api_feeds must appear."
            )
        return self

    @model_validator(mode="after")
    def validate_notable_post_range_coverage(self) -> "ArenaConfig":
        feeds_by_size: dict[str, list[FeedDefinition]] = defaultdict(list)
        for f in self.api_feeds:
            feeds_by_size[f.feed_size].append(f)

        for feed_size, feeds in feeds_by_size.items():
            explicit_langs: set[str] = set()
            for f in feeds:
                if f.included_languages:
                    explicit_langs.update(f.included_languages)
            lang_classes = list(explicit_langs) + ["__other__"]

            for lang_class in lang_classes:
                accepting_feeds: list[FeedDefinition] = []
                for f in feeds:
                    if lang_class == "__other__":
                        if f.included_languages is None:
                            accepting_feeds.append(f)
                    else:
                        if f.included_languages is not None:
                            if lang_class in f.included_languages:
                                accepting_feeds.append(f)
                        elif lang_class not in f.excluded_languages:
                            accepting_feeds.append(f)

                if not accepting_feeds:
                    raise ValueError(
                        f"feed_size '{feed_size}': no feed accepts language class "
                        f"'{lang_class}'. Every language must be routed to at least one feed."
                    )

                ranges = []
                for f in accepting_feeds:
                    if f.notable_post_range is None:
                        ranges.append((0.0, 1.0, f.name))
                    else:
                        ranges.append(
                            (f.notable_post_range[0], f.notable_post_range[1], f.name)
                        )
                ranges.sort()

                expected_start = 0.0
                for floor, ceiling, name in ranges:
                    if floor != expected_start:
                        if floor < expected_start:
                            raise ValueError(
                                f"feed_size '{feed_size}', language '{lang_class}': "
                                f"notable_post_range overlap at {floor} (feed '{name}')"
                            )
                        else:
                            raise ValueError(
                                f"feed_size '{feed_size}', language '{lang_class}': "
                                f"notable_post_range gap between {expected_start} and {floor}"
                            )
                    expected_start = ceiling
                if expected_start != 1.0:
                    raise ValueError(
                        f"feed_size '{feed_size}', language '{lang_class}': "
                        f"notable_post_ranges end at {expected_start}, must reach 1.0"
                    )

        return self


def is_writer_active_for_work_item(
    writer: GrokWriter,
    post_with_context: PostWithContext,
    skip_feed_check: bool = False,
) -> bool:
    if not skip_feed_check:
        feed_match = False
        if post_with_context.api_feed in writer.api_feeds:
            feed_match = True
        elif post_with_context.timed_feed is not None:
            if post_with_context.timed_feed in writer.revision_feeds:
                feed_match = True
            elif post_with_context.timed_feed in writer.retry_feeds:
                feed_match = True
        if not feed_match:
            return False

    if not writer.enabled_ranges:
        return True
    digest = post_digest(
        post_with_context.post.post_id, SUBMISSION_PURPOSE, ALLOCATION_END
    )
    return any(start <= digest < end for start, end in writer.enabled_ranges)


def resolve_model_aliases(config: "ArenaConfig") -> None:
    components = (
        list(config.grok_writers)
        + list(config.rl_rejectors)
        + list(config.screenshot_rejectors)
        + list(config.recent_context_rejectors)
        + list(config.revision_rejectors)
    )
    missing = []
    for component in components:
        env_var = f"MODEL_{component.model_alias.upper()}"
        model_name = os.environ.get(env_var)
        if not model_name:
            missing.append(f"'{component.model_alias}' ({env_var})")
            continue
        component.model_name = model_name
    if missing:
        raise ValueError(
            f"Unresolvable model alias(es): {', '.join(missing)}. "
            "Define the corresponding MODEL_* variables in .env."
        )


def load_and_validate_config(path: str, config_timestamp: int = 0) -> "ArenaConfig":
    with open(path, "rb") as f:
        config_dict = tomllib.load(f)
    config = ArenaConfig(**config_dict, config_timestamp=config_timestamp)
    resolve_model_aliases(config)
    return config
