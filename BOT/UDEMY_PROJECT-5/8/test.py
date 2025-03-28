# Script to verify if Interrupt inherits from BaseException
try:
    from langgraph.types import Interrupt

    # Check if Interrupt is a subclass of BaseException
    if issubclass(Interrupt, BaseException):
        print("✅ Interrupt is correctly defined and inherits from BaseException.")
    else:
        print("❌ Interrupt does not inherit from BaseException.")
except ImportError:
    print("❌ Interrupt could not be imported. Ensure the langgraph library is installed.")
except Exception as e:
    print(f"❌ An error occurred while checking Interrupt: {e}")
