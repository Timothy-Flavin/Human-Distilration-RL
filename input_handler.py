import pygame

def process_events(events, current_mode, text_buffer):
    step_dir = 0
    submitted_note = None
    reset = False
    branch_timeline = False 
    decision = None 
    new_mode = current_mode

    # Handle continuous stepping in 'step' mode
    keys = pygame.key.get_pressed()
    if current_mode == "step":
        if keys[pygame.K_RIGHT]:
            step_dir = 1
        elif keys[pygame.K_LEFT]:
            step_dir = -1

    for event in events:
        if event.type == pygame.QUIT:
            return "quit", text_buffer, None, 0, False, False, None

        # --- TEXT INPUT MODE ---
        if current_mode == "note":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    submitted_note = text_buffer
                    new_mode = "step"
                    text_buffer = ""
                elif event.key == pygame.K_ESCAPE:
                    new_mode = "step"
                    text_buffer = ""
                elif event.key == pygame.K_BACKSPACE:
                    text_buffer = text_buffer[:-1]
            elif event.type == pygame.TEXTINPUT:
                text_buffer += event.text
        
        # --- DECISION MODE (Accept/Reject Override) ---
        elif current_mode == "decision":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    decision = "accept"
                    new_mode = "step"
                elif event.key == pygame.K_r:
                    decision = "reject"
                    new_mode = "step"

        # --- PLAYBACK / CONTROL MODES ---
        else:
            if event.type == pygame.KEYDOWN:
                # Toggle Realtime
                if event.key == pygame.K_SPACE:
                    if current_mode != "realtime":
                        new_mode = "realtime"
                        branch_timeline = True  # We are taking control!
                    else:
                        new_mode = "decision" # Transition to decision
                        
                # Toggle Agent
                elif event.key == pygame.K_TAB:
                    if current_mode != "agent":
                        new_mode = "agent"
                        branch_timeline = True  # Agent is taking control!
                    else:
                        new_mode = "decision" # Transition to decision
                        
                elif event.key == pygame.K_RETURN:
                    new_mode = "note"
                    text_buffer = ""
                elif event.key == pygame.K_r:
                    reset = True
                elif event.key == pygame.K_q:
                    new_mode = "finish"

                # Stepping
                # (Stepping is now handled continuously above)
                pass

    return new_mode, text_buffer, submitted_note, step_dir, reset, branch_timeline, decision

def get_realtime_action(keys):
    """
    Maps continuous key presses to environment actions.
    Note: Customize this mapping based on your specific Gymnasium environment.
    Example mapping for discrete environments (e.g., LunarLander):
    0: NOOP, 1: LEFT, 2: MAIN ENGINE, 3: RIGHT
    """
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 1
    elif keys[pygame.K_UP] or keys[pygame.K_w]:
        return 2
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 3
    return 0  # NOOP