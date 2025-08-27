def round_to_fit(n, digits):
    """
    Rounds as little as possible for it to fit in a certain number of digits.
    In the most extreme case, it returns n with all its decimal places cut off. 
    """
    assert int(digits) == digits and digits > 0, "Round up to a positive integer number of digits."
    integer_digits = num_digits = len(str(int(abs(n))))
    
    if integer_digits >= digits: # Returns the nearest integer if we don't have any space for any decimal places
        return int(round(n,0))
    
    return round(n, digits-integer_digits) # Returns the rational number with as many digits as it can to fit