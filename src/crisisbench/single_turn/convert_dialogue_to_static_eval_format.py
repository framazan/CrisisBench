import pandas as pd
import argparse
import uuid
import random
from typing import List, Tuple

def strip_counselor_prefix(message: str) -> str:
    """
    Remove the 'Counselor:' prefix from every line in a (possibly multi-line)
    counselor block.

    Args:
        message: The counselor message block, potentially spanning multiple lines
                 each starting with 'Counselor:'

    Returns:
        Message content with all 'Counselor:' prefixes stripped, joined back
        into a single string.
    """
    prefix = "Counselor:"
    stripped_lines = []
    for line in message.split("\n"):
        if line.startswith(prefix):
            stripped_lines.append(line[len(prefix):].strip())
        else:
            stripped_lines.append(line)
    return "\n".join(stripped_lines).strip()

def split_conversation(conversation: str) -> List[Tuple[str, str]]:
    """
    Split a conversation into pairs of (conversation_history, counselor_message).

    Rather than treating each newline as a message boundary, we first group
    consecutive lines into *speaker blocks* — a new block begins whenever a
    line starts with a recognised speaker prefix (Counselor:, Patient:, etc.).
    A (history, counselor_message) pair is emitted after each complete
    counselor block, with all preceding blocks forming the history.

    Args:
        conversation: String containing the full conversation

    Returns:
        List of tuples (conversation_history, counselor_message)
    """
    SPEAKER_PREFIXES = ("Counselor:", "Patient:", "Therapist:", "Client:")

    def get_speaker(line: str) -> str:
        """Return the speaker prefix that starts this line, or empty string."""
        for prefix in SPEAKER_PREFIXES:
            if line.startswith(prefix):
                return prefix
        return ""

    lines = conversation.split('\n')

    # ── Group lines into speaker blocks ──────────────────────────────────────
    # A new block is started only when the SPEAKER CHANGES, so consecutive
    # "Counselor: ..." lines all belong to the same counselor block.
    blocks: List[str] = []
    current_lines: List[str] = []
    current_speaker: str = ""

    for line in lines:
        speaker = get_speaker(line)
        if speaker and speaker != current_speaker:
            # Speaker has changed — flush the previous block and start a new one
            if current_lines:
                blocks.append('\n'.join(current_lines))
            current_lines = [line]
            current_speaker = speaker
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append('\n'.join(current_lines))

    # ── Emit one pair per counselor block ────────────────────────────────────
    result = []
    for i, block in enumerate(blocks):
        if block.lstrip().startswith('Counselor:'):
            history = '\n'.join(blocks[:i]) if i > 0 else ''
            result.append((history, block.strip()))

    return result

def filter_counselor_messages(message_pairs: List[Tuple[str, str]], n_messages: int) -> List[Tuple[str, str]]:
    """
    Filter out the first and last n counselor messages while preserving conversation history.
    
    Args:
        message_pairs: List of (history, counselor_message) tuples
        n_messages: Number of messages to remove from start and end
        
    Returns:
        Filtered list of (history, counselor_message) tuples
    """
    if len(message_pairs) <= 2 * n_messages:
        # If there aren't enough messages after filtering, return empty list
        return []
    
    # Return messages excluding first and last n
    return message_pairs[n_messages:-n_messages]

def process_dialogues(input_csv: str, output_csv: str, conversation_column: str, n_filter_messages: int = 5, max_messages_per_convo: int = None):
    """
    Process dialogues from input CSV and create a new CSV with one row per counselor message,
    excluding the first and last n counselor messages.
    
    Args:
        input_csv: Path to input CSV containing dialogues
        output_csv: Path to save the processed CSV
        conversation_column: Name of the column containing the conversation text
        n_filter_messages: Number of messages to filter from start and end of each conversation
        max_messages_per_convo: Max messages to select per conversation
    """
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    if conversation_column not in df.columns:
        raise ValueError(f"Column '{conversation_column}' not found in input CSV. Available columns: {', '.join(df.columns)}")
    
    # List to store new rows
    rows = []
    
    # Random number generator with seed 16
    rng = random.Random(16)
    
    # Process each conversation
    for idx, row in df.iterrows():
        conversation = row[conversation_column]
        conversation_id = row.get('uid', str(idx))  # Use provided conversation_id or create one
        
        # Split conversation into history-message pairs
        message_pairs = split_conversation(conversation)
        
        # Filter out first and last n messages
        filtered_pairs = filter_counselor_messages(message_pairs, n_filter_messages)
        if max_messages_per_convo:
            # Sample max_messages_per_convo messages randomly
            if len(filtered_pairs) > max_messages_per_convo:
                filtered_pairs = rng.sample(filtered_pairs, max_messages_per_convo)
            
        # Create a row for each remaining counselor message
        for history, counselor_msg in filtered_pairs:
            rows.append({
                'uid': str(uuid.uuid4()),  # Generate unique identifier
                'conversation_id': conversation_id,
                'convo_history': history,
                'counselor_message': strip_counselor_prefix(counselor_msg)
            })
    
    # Create new dataframe and save
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Processed {len(df)} conversations into {len(rows)} counselor messages")
    print(f"Note: First and last {n_filter_messages} counselor messages were excluded from each conversation")

def main():
    parser = argparse.ArgumentParser(description='Process dialogues into individual counselor messages')
    parser.add_argument('input_csv', help='Path to input CSV file containing dialogues')
    parser.add_argument('output_csv', help='Path to save the processed CSV file')
    parser.add_argument('--conversation-column', '-c', default='conversation',
                      help='Name of the column containing the conversation text (default: conversation)')
    parser.add_argument('--filter-messages', '-n', type=int, default=5,
                      help='Number of counselor messages to remove from start and end of each conversation (default: 5)')
    parser.add_argument('--max-messages-per-convo', '-m', type=int, default = None,
                      help='Maximum number of messages to select randomly per conversation')
    
    args = parser.parse_args()
    process_dialogues(args.input_csv, args.output_csv, args.conversation_column, args.filter_messages, args.max_messages_per_convo)

if __name__ == '__main__':
    main() 