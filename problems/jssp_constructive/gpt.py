def determine_next_operation(current_status, feasible_operations):
    if not feasible_operations:
        return None
    # Select the operation with the shortest processing time
    result = min(feasible_operations, key=lambda op: op['processing_time'])
    return result
