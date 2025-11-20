"""
Quick test to verify ODE solver implementation works correctly.
Run this to check that everything is properly integrated.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.inference.ode_solvers import (
            ODEFlowSolver,
            sample_with_ode_solver,
            compare_solvers,
            VelocityField
        )
        print("✓ ode_solvers module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_torchdiffeq():
    """Test that torchdiffeq is installed."""
    print("\nTesting torchdiffeq...")
    try:
        import torchdiffeq
        print(f"✓ torchdiffeq version {torchdiffeq.__version__} installed")
        return True
    except ImportError:
        print("✗ torchdiffeq not installed. Run: pip install torchdiffeq")
        return False


def test_basic_functionality():
    """Test basic ODE solver functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from src.inference.ode_solvers import ODEFlowSolver, sample_with_ode_solver
        from src.models.mlp import MLP
        
        # Create simple model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MLP(input_dim=64, hidden_dim=128)
        model.eval()
        model = model.to(device)
        
        # Create test input
        condition = torch.randn(2, 8, 8).to(device)
        
        # Test with different solvers
        solvers_to_test = ['euler', 'rk4', 'dopri5']
        
        for solver_name in solvers_to_test:
            try:
                samples = sample_with_ode_solver(
                    model=model,
                    condition=condition,
                    solver=solver_name,
                    n_steps=10,
                    device=device
                )
                assert samples.shape == condition.shape, f"Shape mismatch for {solver_name}"
                print(f"  ✓ {solver_name} solver works")
            except Exception as e:
                print(f"  ✗ {solver_name} solver failed: {e}")
                return False
        
        # Test trajectory
        samples, trajectory = sample_with_ode_solver(
            model=model,
            condition=condition,
            solver='dopri5',
            return_trajectory=True,
            device=device
        )
        assert trajectory.shape[0] > 1, "Trajectory should have multiple time steps"
        print(f"  ✓ Trajectory generation works ({trajectory.shape[0]} steps)")
        
        # Test ODEFlowSolver class
        solver = ODEFlowSolver(
            model=model,
            solver='dopri5',
            rtol=1e-5,
            atol=1e-7
        )
        samples = solver.sample(condition=condition)
        assert samples.shape == condition.shape
        print("  ✓ ODEFlowSolver class works")
        
        print("\n✓ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_integration():
    """Test integration with existing inference code."""
    print("\nTesting inference integration...")
    
    try:
        from inference.inference import sample_flow_matching
        from src.models.mlp import MLP
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = MLP(input_dim=64, hidden_dim=128)
        model.eval()
        model = model.to(device)
        
        condition = torch.randn(2, 8, 8).to(device)
        
        # Test with ODE solver through existing interface
        samples = sample_flow_matching(
            model=model,
            condition=condition,
            integration_method='dopri5',
            rtol=1e-5,
            atol=1e-7,
            device=device
        )
        
        assert samples.shape == condition.shape
        print("  ✓ Integration with sample_flow_matching works")
        
        # Test with old methods (backward compatibility)
        samples_euler = sample_flow_matching(
            model=model,
            condition=condition,
            integration_method='euler',
            n_steps=50,
            device=device
        )
        
        assert samples_euler.shape == condition.shape
        print("  ✓ Backward compatibility maintained")
        
        print("\n✓ Inference integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Inference integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("ODE SOLVERS - INTEGRATION TEST")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_import()))
    
    # Test torchdiffeq
    results.append(("torchdiffeq", test_torchdiffeq()))
    
    # Only run functionality tests if imports work
    if results[0][1] and results[1][1]:
        results.append(("Basic Functionality", test_basic_functionality()))
        results.append(("Inference Integration", test_inference_integration()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou're ready to use ODE solvers!")
        print("\nNext steps:")
        print("  1. Run: python examples/ode_solver_demo.py")
        print("  2. Try: jupyter notebook notebooks/ode_solver_tutorial.ipynb")
        print("  3. Read: docs/ODE_SOLVERS_GUIDE.md")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        if not results[1][1]:  # torchdiffeq not installed
            print("\nMake sure to install torchdiffeq:")
            print("  pip install torchdiffeq")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
