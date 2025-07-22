# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

"""
Utility functions, HTTP helpers, response formatting, and exception handling for ServerlessLLM.
"""

import asyncio
import random
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from sllm.serve.logger import init_logger

logger = init_logger(__name__)

# =============================================================================
# Name Generation
# =============================================================================

POSITIONS = [
    "director",
    "manager",
    "assistant",
    "analyst",
    "engineer",
    "designer",
    "developer",
    "coordinator",
    "specialist",
    "consultant",
    "supervisor",
    "administrator",
    "technician",
    "operator",
    "inspector",
    "architect",
    "strategist",
    "planner",
    "researcher",
    "scientist",
    "programmer",
    "tester",
    "writer",
    "editor",
    "artist",
    "photographer",
    "translator",
    "recruiter",
    "trainer",
    "auditor",
    "advisor",
    "commander",
    "lieutenant",
    "sergeant",
    "captain",
    "pilot",
    "navigator",
    "mechanic",
    "clerk",
    "officer",
    "associate",
    "executive",
    "lead",
    "senior",
    "junior",
    "chief",
    "head",
    "vice",
    "deputy",
    "principal",
    "staff",
    "intern",
    "agent",
    "operative",
    "handler",
    "guard",
    "broker",
    "trader",
    "vendor",
    "courier",
    "messenger",
    "trainer",
    "magi",
    "watchman",
    "porter",
    "janitor",
    "custodian",
    "maintenance",
    "fabricator",
    "assembler",
    "nurse",
    "doctor",
    "therapist",
    "counselor",
    "teacher",
    "professor",
    "student",
    "scholar",
    "librarian",
    "archivist",
    "cook",
    "chef",
    "waiter",
    "barista",
    "server",
    "host",
    "receptionist",
    "ace",
    "aroma",
    "artist",
    "beauty",
    "biker",
    "bird",
    "blackbelt",
    "boarder",
    "bodybuilder",
    "bugcatcher",
    "burglar",
    "camper",
    "channeler",
    "collector",
    "cooltrainer",
    "crusher",
    "cyclist",
    "dancer",
    "delinquent",
    "diver",
    "dragontamer",
    "electrician",
    "expert",
    "fisherman",
    "gambler",
    "gentleman",
    "guitarist",
    "harlequin",
    "hiker",
    "juggler",
    "kindler",
    "lady",
    "lass",
    "leader",
    "madame",
    "maid",
    "musician",
    "ninja",
    "oracle",
    "painter",
    "parasol",
    "picnicker",
    "policeman",
    "preschooler",
    "psychic",
    "punk",
    "rancher",
    "rival",
    "roughneck",
    "ruinmaniac",
    "sailor",
    "schoolkid",
    "skier",
    "striker",
    "swimmer",
    "tamer",
    "teammate",
    "triathlete",
    "tuber",
    "veteran",
    "waitress",
    "worker",
    "youngster",
    "elite",
    "champion",
    "hexmaniac",
    "advanced",
    "master",
    "novice",
    "skilled",
    "prime",
    "supreme",
    "ultimate",
    "legendary",
    "mythic",
    "epic",
    "rare",
    "common",
    "basic",
    "standard",
    "premium",
    "deluxe",
    "special",
    "unique",
    "custom",
    "tactical",
    "strategic",
    "covert",
    "stealth",
    "shadow",
    "ghost",
    "phantom",
    "void",
    "cyber",
    "digital",
    "quantum",
    "nuclear",
    "atomic",
    "stellar",
    "cosmic",
    "solar",
    "lunar",
    "astral",
    "ethereal",
    "spectral",
    "temporal",
    "dimensional",
    "spatial",
    "crimson",
    "azure",
    "golden",
    "silver",
    "platinum",
    "diamond",
    "crystal",
    "obsidian",
    "emerald",
    "ruby",
    "sapphire",
    "onyx",
    "amber",
    "jade",
    "pearl",
    "opal",
    "swift",
    "rapid",
    "quick",
    "fast",
    "slow",
    "steady",
    "calm",
    "fierce",
    "wild",
    "tame",
    "gentle",
    "harsh",
    "soft",
    "hard",
    "light",
    "dark",
    "bright",
    "dim",
    "loud",
    "quiet",
    "hot",
    "cold",
    "warm",
    "cool",
    "fresh",
    "stale",
    "new",
    "old",
    "young",
    "ancient",
    "modern",
    "classic",
    "bold",
    "shy",
    "brave",
    "timid",
    "strong",
    "weak",
    "tough",
    "fragile",
    "smart",
    "wise",
    "clever",
    "dull",
    "sharp",
    "blunt",
    "smooth",
    "rough",
    "heavy",
    "big",
    "small",
    "huge",
    "tiny",
    "giant",
    "mini",
    "long",
    "short",
    "tall",
    "wide",
    "narrow",
    "thick",
    "thin",
    "deep",
    "shallow",
    "high",
    "low",
    "up",
    "down",
    "left",
    "right",
    "center",
    "inner",
    "outer",
    "hidden",
    "visible",
    "secret",
    "open",
    "closed",
    "locked",
    "free",
    "bound",
    "loose",
    "tight",
    "slack",
    "tense",
    "relaxed",
    "stressed",
    "active",
    "passive",
    "dynamic",
    "static",
    "mobile",
    "fixed",
    "fluid",
    "solid",
    "rickety",
    "jetstream",
    "bizarre",
    "funny",
    "father",
    "mother",
    "spider",
    "iron",
    "super",
    "aqua",
    "wonder",
    "blind",
    "holy",
]

NAMES = [
    "shinji",
    "rei",
    "asuka",
    "misato",
    "ikari",
    "kaworu",
    "balthasar",
    "caspar",
    "melchior",
    "goku",
    "vegeta",
    "gohan",
    "piccolo",
    "frieza",
    "pascal",
    "turing",
    "ampere",
    "lovelace",
    "hopper",
    "simon",
    "kamina",
    "yoko",
    "nia",
    "viral",
    "jonathan",
    "jotaro",
    "dio",
    "joseph",
    "josuke",
    "giorno",
    "jolyne",
    "johnny",
    "gappy",
    "jodio",
    "kars",
    "wamuu",
    "esidisi",
    "kira",
    "diavolo",
    "pucci",
    "valentine",
    "tooru",
    "steel",
    "john",
    "winston",
    "charon",
    "sofia",
    "caine",
    "okabe",
    "kurisu",
    "mayuri",
    "daru",
    "suzuha",
    "christina",
    "ash",
    "misty",
    "brock",
    "may",
    "dawn",
    "serena",
    "clemont",
    "bonnie",
    "cilan",
    "iris",
    "tracey",
    "max",
    "lana",
    "kiawe",
    "lillie",
    "sophocles",
    "mallow",
    "goh",
    "chloe",
    "gary",
    "paul",
    "drew",
    "kenny",
    "zoey",
    "trip",
    "bianca",
    "stephan",
    "cameron",
    "virgil",
    "tierno",
    "trevor",
    "shauna",
    "sawyer",
    "alain",
    "mairin",
    "gladion",
    "hau",
    "acerola",
    "professor",
    "oak",
    "elm",
    "birch",
    "rowan",
    "juniper",
    "sycamore",
    "kukui",
    "burnet",
    "cerise",
    "magnolia",
    "sonia",
    "jacq",
    "nemona",
    "jessie",
    "james",
    "meowth",
    "giovanni",
    "archer",
    "ariana",
    "petrel",
    "proton",
    "silver",
    "lance",
    "karen",
    "will",
    "koga",
    "bruno",
    "lorelei",
    "agatha",
    "sidney",
    "phoebe",
    "glacia",
    "drake",
    "aaron",
    "bertha",
    "flannery",
    "lucian",
    "cynthia",
    "alder",
    "diantha",
    "leon",
    "nico",
    "andy",
    "victor",
    "juiz",
    "top",
    "shen",
    "feng",
    "mui",
    "steven",
    "fuuko",
    "barry",
    "ethan",
    "bocchi",
    "kikuri",
    "seika",
    "loid",
    "yor",
    "anya",
    "handler",
    "fiona",
    "franky",
    "denji",
    "aki",
    "power",
    "makima",
    "reze",
    "quanxi",
    "kobeni",
    "himeno",
    "kishibe",
    "yoshida",
    "chihiro",
    "shiba",
    "hakuri",
    "samura",
    "dennis",
    "frank",
    "dee",
    "mac",
    "cricket",
    "sam",
    "alan",
    "ellie",
    "ian",
    "lex",
    "tim",
    "ray",
    "henry",
    "martin",
    "sarah",
    "roland",
    "peter",
    "nick",
    "kelly",
    "eddie",
    "dieter",
    "carter",
    "robert",
    "amanda",
    "billy",
    "udesky",
    "cooper",
    "nash",
    "mark",
    "bruce",
    "clark",
    "diana",
    "barry",
    "arthur",
    "hal",
    "oliver",
    "dinah",
    "kara",
    "barbara",
    "dick",
    "jason",
    "damian",
    "stephanie",
    "cassandra",
    "kate",
    "helena",
    "selina",
    "pamela",
    "harvey",
    "edward",
    "oswald",
    "jarvis",
    "roman",
    "floyd",
    "harley",
    "joker",
    "lex",
    "doomsday",
    "darkseid",
    "brainiac",
    "sinestro",
    "atrocitus",
    "larfleeze",
    "saint",
    "zatanna",
    "raven",
    "starfire",
    "cyborg",
    "beast",
    "terra",
    "blue",
    "booster",
    "ted",
    "peter",
    "tony",
    "steve",
    "natasha",
    "clint",
    "thor",
    "loki",
    "scott",
    "hope",
    "wanda",
    "vision",
    "james",
    "bucky",
    "rhodey",
    "pepper",
    "happy",
    "may",
    "ned",
    "michelle",
    "flash",
    "gwen",
    "miles",
    "miguel",
    "logan",
    "ororo",
    "kurt",
    "kitty",
    "bobby",
    "warren",
    "jean",
    "erik",
    "charles",
    "remy",
    "anna",
    "betsy",
    "emma",
    "hank",
    "reed",
    "sue",
    "ben",
    "franklin",
    "valeria",
    "stephen",
    "wong",
    "ancient",
    "mordo",
    "carol",
    "monica",
    "kamala",
    "matt",
    "jessica",
    "luke",
    "danny",
    "wade",
    "cable",
    "domino",
    "colossus",
    "deadpool",
    "gambit",
    "rogue",
    "storm",
    "wolverine",
    "cyclops",
    "phoenix",
    "america",
    "man",
    "woman",
    "bandit",
    "eren",
    "mikasa",
    "armin",
    "levi",
    "erwin",
    "hange",
    "sasha",
    "connie",
    "jean",
    "marco",
    "annie",
    "reiner",
    "bertholdt",
    "ymir",
    "historia",
    "christa",
    "zeke",
    "grisha",
    "carla",
    "keith",
    "shadis",
    "pixis",
    "nile",
    "dok",
    "rico",
    "mitabi",
    "thomas",
    "mina",
    "samuel",
    "daz",
    "marlowe",
    "hitch",
    "boris",
    "gordon",
    "sandra",
    "mylius",
    "jurgen",
    "ivan",
    "petra",
    "oluo",
    "eld",
    "gunther",
    "moblit",
    "nanaba",
    "gelgar",
    "lynne",
    "henning",
    "tomas",
    "klaus",
    "marlene",
    "lauda",
    "franz",
    "hannah",
    "gustav",
    "anka",
    "hugo",
    "rashad",
    "aphrodite",
    "artemis",
    "zeus",
    "poseidon",
    "hades",
    "hera",
    "hephaestus",
    "ares",
    "hermes",
    "dionysus",
    "hestia",
    "apollo",
    "athena",
    "demeter",
    "zagreus",
    "megaera",
    "sisyphus",
    "thanatos",
    "melinoe",
    "nyx",
    "hecate",
    "mipha",
    "revali",
    "daruk",
    "urbosa",
    "link",
    "zelda",
    "sidon",
]


def generate_name():
    """Generate a random worker name in the format 'position-name'"""
    position = random.choice(POSITIONS)
    name = random.choice(NAMES)
    return f"{position}-{name}"


# =============================================================================
# Response Formatting Utilities
# =============================================================================


def success_response(
    message: str, data: Optional[Any] = None, status: str = "ok"
) -> Dict[str, Any]:
    """
    Create a standardized success response.

    Args:
        message: Success message
        data: Optional response data
        status: Status string (default: "ok")

    Returns:
        Standardized success response dictionary
    """
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if data is not None:
        response["data"] = data

    return response


def operation_response(
    operation: str,
    resource: str,
    resource_id: Optional[str] = None,
    data: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create a standardized operation response.

    Args:
        operation: The operation performed (e.g., "created", "updated", "deleted")
        resource: The resource type (e.g., "model", "worker", "task")
        resource_id: Optional resource identifier
        data: Optional response data

    Returns:
        Standardized operation response dictionary
    """
    message = f"{resource.title()} {operation} successfully"
    if resource_id:
        message += f": {resource_id}"

    return success_response(message=message, data=data)


def task_response(task_id: str, data: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create a standardized task response.

    Args:
        task_id: The task identifier
        data: Optional task result data

    Returns:
        Standardized task response dictionary
    """
    return success_response(
        message=f"Task {task_id} completed successfully",
        data={"task_id": task_id, "result": data},
    )


def list_response(
    items: list, resource_type: str, total_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a standardized list response.

    Args:
        items: List of items
        resource_type: Type of resources being listed
        total_count: Optional total count (if different from len(items))

    Returns:
        Standardized list response dictionary
    """
    count = total_count if total_count is not None else len(items)

    return success_response(
        message=f"Retrieved {count} {resource_type}(s)",
        data={"items": items, "count": count},
    )


def health_response(
    services: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized health check response.

    Args:
        services: Optional dict of service name -> status

    Returns:
        Standardized health response dictionary
    """
    data = {"healthy": True}
    if services:
        data["services"] = services

    return success_response(message="Service is healthy", data=data)


# =============================================================================
# Exception Hierarchy
# =============================================================================


class ServerlessLLMError(Exception):
    """Base exception for all ServerlessLLM errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ValidationError(ServerlessLLMError):
    """Raised when input validation fails."""

    pass


class ResourceNotFoundError(ServerlessLLMError):
    """Raised when a requested resource cannot be found."""

    pass


class ResourceConflictError(ServerlessLLMError):
    """Raised when a resource operation conflicts with current state."""

    pass


class InternalServerError(ServerlessLLMError):
    """Raised for internal server errors."""

    pass


class WorkerError(ServerlessLLMError):
    """Raised for worker-related errors."""

    pass


class ModelError(ServerlessLLMError):
    """Raised for model-related errors."""

    pass


class TaskError(ServerlessLLMError):
    """Raised for task processing errors."""

    pass


class RedisError(ServerlessLLMError):
    """Raised for Redis connection/operation errors."""

    pass


class TimeoutError(ServerlessLLMError):
    """Raised when an operation times out."""

    pass


class HTTPRetryError(Exception):
    """Raised when HTTP request fails after all retries."""

    pass


def standardize_error_response(error: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format.

    Args:
        error: The exception to convert

    Returns:
        Standardized error response dictionary
    """
    if isinstance(error, ServerlessLLMError):
        return {
            "error": {
                "code": error.error_code,
                "message": error.message,
                "details": error.details,
            }
        }
    else:
        # Handle non-ServerlessLLM exceptions
        return {
            "error": {
                "code": "InternalError",
                "message": str(error),
                "details": {},
            }
        }


def map_to_http_status(error: Exception) -> int:
    """
    Map exception types to appropriate HTTP status codes.

    Args:
        error: The exception to map

    Returns:
        HTTP status code
    """
    if isinstance(error, ValidationError):
        return 400
    elif isinstance(error, ResourceNotFoundError):
        return 404
    elif isinstance(error, ResourceConflictError):
        return 409
    elif isinstance(error, TimeoutError):
        return 408
    elif isinstance(error, (WorkerError, ModelError, TaskError)):
        return 503  # Service Unavailable
    elif isinstance(error, RedisError):
        return 503  # Service Unavailable
    elif isinstance(error, InternalServerError):
        return 500
    else:
        return 500  # Default to internal server error


# =============================================================================
# HTTP Utilities
# =============================================================================


async def http_request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    timeout: float = 30.0,
    **kwargs,
) -> aiohttp.ClientResponse:
    """
    Make HTTP request with exponential backoff retry logic.

    Args:
        session: aiohttp ClientSession
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to session.request()

    Returns:
        aiohttp.ClientResponse object

    Raises:
        HTTPRetryError: If all retries are exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            kwargs.setdefault("timeout", timeout_obj)

            async with session.request(method, url, **kwargs) as response:
                # Consider 5xx status codes as retryable errors
                if response.status >= 500:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Server error: {response.status}",
                    )
                return response

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    f"HTTP request failed after {max_retries + 1} attempts to {url}: {e}"
                )
                break

            # Calculate exponential backoff delay
            delay = min(base_delay * (2**attempt), max_delay)
            logger.warning(
                f"HTTP request to {url} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)

    raise HTTPRetryError(
        f"HTTP request to {url} failed after {max_retries + 1} attempts: {last_exception}"
    )


async def post_json_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    timeout: float = 30.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    POST JSON payload with retry logic and return JSON response.

    Args:
        session: aiohttp ClientSession
        url: Request URL
        payload: JSON payload to send
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to http_request_with_retry()

    Returns:
        Parsed JSON response as dict

    Raises:
        HTTPRetryError: If all retries are exhausted
        ValueError: If response is not valid JSON
    """
    kwargs.setdefault("json", payload)

    response = await http_request_with_retry(
        session=session,
        method="POST",
        url=url,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )

    try:
        return await response.json()
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response from {url}: {e}")


async def get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
    timeout: float = 30.0,
    **kwargs,
) -> aiohttp.ClientResponse:
    """
    GET request with retry logic.

    Args:
        session: aiohttp ClientSession
        url: Request URL
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to http_request_with_retry()

    Returns:
        aiohttp.ClientResponse object

    Raises:
        HTTPRetryError: If all retries are exhausted
    """
    return await http_request_with_retry(
        session=session,
        method="GET",
        url=url,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )
