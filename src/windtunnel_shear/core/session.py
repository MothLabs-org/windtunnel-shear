"""Session recorder and episode grouping logic."""

from __future__ import annotations

from datetime import datetime, timezone

from windtunnel_shear.core.models import (
    Episode,
    InterceptedRequest,
    InterceptedResponse,
    Session,
    Turn,
    TurnType,
)


def _classify_turn(
    request: InterceptedRequest,
    response: InterceptedResponse,
) -> TurnType:
    """Classify a turn based on the request and response content."""
    if request.is_tool_result:
        return TurnType.TOOL_RESULT_SUBMISSION
    if response.has_tool_calls:
        return TurnType.TOOL_CALL_REQUEST
    if response.finish_reason == "stop":
        return TurnType.FINAL_RESPONSE
    return TurnType.USER_MESSAGE


class SessionRecorder:
    """Records intercepted traffic into a Session with episode grouping.

    The recorder accumulates turns and groups them into episodes based on
    conversation boundaries (e.g. a new user message after a final response).
    """

    def __init__(self) -> None:
        self._session = Session(
            started_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._current_episode: Episode | None = None
        self._last_turn_type: TurnType | None = None

    @property
    def session(self) -> Session:
        """Return the current session being recorded."""
        return self._session

    def record_turn(
        self,
        request: InterceptedRequest,
        response: InterceptedResponse,
    ) -> None:
        """Record a single request/response turn.

        Args:
            request: The intercepted request.
            response: The intercepted response.
        """
        turn_type = _classify_turn(request, response)

        # Start a new episode if needed
        if self._should_start_new_episode(turn_type):
            self._current_episode = Episode()
            self._session.episodes.append(self._current_episode)

        if self._current_episode is None:
            self._current_episode = Episode()
            self._session.episodes.append(self._current_episode)

        # Compute request hash for replay matching
        request.compute_hash()

        # Set model from first request if not already set
        if not self._session.model and request.model:
            self._session.model = request.model

        turn = Turn(
            turn_type=turn_type,
            request=request,
            response=response,
        )
        self._current_episode.turns.append(turn)
        self._last_turn_type = turn_type

    def _should_start_new_episode(self, turn_type: TurnType) -> bool:
        """Decide if a new episode should start."""
        if self._current_episode is None:
            return True
        # New episode after a final response, unless it's a tool result
        return (
            self._last_turn_type == TurnType.FINAL_RESPONSE
            and turn_type != TurnType.TOOL_RESULT_SUBMISSION
        )

    def finalize(self) -> Session:
        """Finalize and return the completed session.

        Returns:
            The recorded session with all episodes closed.
        """
        return self._session
