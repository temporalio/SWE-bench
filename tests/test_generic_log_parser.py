import pytest
from swebench.harness.grading import parse_log_generic_exit_codes
from swebench.harness.constants import TestStatus


class TestParseLogGenericExitCodes:
    """Tests for the generic fallback parser that handles exit codes and step declarations."""

    def test_exit_codes_with_step_declarations(self):
        """Test parsing with step declarations and some skipped steps."""
        log = """
        Starting test suite
        + : '>>>>> EXPECTED_STEPS: setup,typecheck,unit_tests,integration_tests,cleanup'
        + : '>>>>> setup: 0'
        + : '>>>>> typecheck: 0'
        + : '>>>>> unit_tests: 1'
        Unit tests failed, skipping integration tests
        + : '>>>>> cleanup: 0'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "setup": TestStatus.PASSED.value,
            "typecheck": TestStatus.PASSED.value,
            "unit_tests": TestStatus.FAILED.value,
            "integration_tests": TestStatus.SKIPPED.value,  # Declared but not executed
            "cleanup": TestStatus.PASSED.value,
        }
        assert result == expected

    def test_multiple_step_declarations(self):
        """Test handling multiple step declaration lines."""
        log = """
        + : '>>>>> EXPECTED_STEPS: step1,step2'
        Some output
        + : '>>>>> EXPECTED_STEPS: step3,step4,step5'
        + : '>>>>> step1: 0'
        + : '>>>>> step3: 1'
        + : '>>>>> step5: 0'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "step1": TestStatus.PASSED.value,
            "step2": TestStatus.SKIPPED.value,  # From first declaration
            "step3": TestStatus.FAILED.value,
            "step4": TestStatus.SKIPPED.value,  # From second declaration
            "step5": TestStatus.PASSED.value,
        }
        assert result == expected

    def test_complex_test_names(self):
        """Test parsing with complex test names containing underscores and numbers."""
        log = """
        + : '>>>>> EXPECTED_STEPS: test_setup_phase_1,test_module_123,complex_test_name_with_underscores'
        + : '>>>>> test_setup_phase_1: 0'
        + : '>>>>> test_module_123: 1'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "test_setup_phase_1": TestStatus.PASSED.value,
            "test_module_123": TestStatus.FAILED.value,
            "complex_test_name_with_underscores": TestStatus.SKIPPED.value,
        }
        assert result == expected

    def test_mixed_log_content_with_noise(self):
        """Test parsing with lots of irrelevant log content."""
        log = """
        2025-10-03 12:23:12,866 - INFO - Starting test
        Error: some unrelated error
        + cd /testbed
        + git config --global --add safe.directory /testbed
        + : '>>>>> EXPECTED_STEPS: build,test,deploy'
        Building project...
        Build successful
        + : '>>>>> build: 0'
        Running tests...
        Test failed with error
        + : '>>>>> test: 1'
        Skipping deployment due to test failure
        + echo 'Cleanup complete'
        Cleanup complete
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "build": TestStatus.PASSED.value,
            "test": TestStatus.FAILED.value,
            "deploy": TestStatus.SKIPPED.value,
        }
        assert result == expected

    def test_empty_log(self):
        """Test parsing an empty log."""
        log = ""
        result = parse_log_generic_exit_codes(log, None)
        assert result == {}

    def test_log_with_no_matching_patterns(self):
        """Test log with no exit codes or step declarations."""
        log = """
        Some random log output
        Error messages
        Build output
        No test markers
        """
        
        result = parse_log_generic_exit_codes(log, None)
        assert result == {}

    def test_malformed_exit_codes(self):
        """Test handling of malformed exit code lines."""
        log = """
        + : '>>>>> EXPECTED_STEPS: test1,test2,test3'
        + : '>>>>> test1: 0'
        + : '>>>>> test2: not_a_number'
        + : '>>>>> test3: 1'
        + : '>>>>> malformed_line_no_colon'
        + : '>>>>> : 0'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        # Only valid exit codes should be parsed
        expected = {
            "test1": TestStatus.PASSED.value,
            "test2": TestStatus.SKIPPED.value,  # Declared but malformed exit code
            "test3": TestStatus.FAILED.value,
        }
        assert result == expected

    def test_empty_step_declarations(self):
        """Test handling of empty or malformed step declarations."""
        log = """
        + : '>>>>> EXPECTED_STEPS: '
        + : '>>>>> EXPECTED_STEPS: ,,,'
        + : '>>>>> EXPECTED_STEPS: step1,,step2,'
        + : '>>>>> step1: 0'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "step1": TestStatus.PASSED.value,
            "step2": TestStatus.SKIPPED.value,
        }
        assert result == expected

    def test_non_zero_exit_codes(self):
        """Test various non-zero exit codes all map to FAILED."""
        log = """
        + : '>>>>> EXPECTED_STEPS: test1,test2,test3,test4,test5'
        + : '>>>>> test1: 1'
        + : '>>>>> test2: 2'
        + : '>>>>> test3: 127'
        + : '>>>>> test4: 255'
        + : '>>>>> test5: 0'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "test1": TestStatus.FAILED.value,
            "test2": TestStatus.FAILED.value,
            "test3": TestStatus.FAILED.value,
            "test4": TestStatus.FAILED.value,
            "test5": TestStatus.PASSED.value,
        }
        assert result == expected

    def test_case_sensitivity(self):
        """Test that step names are case sensitive."""
        log = """
        + : '>>>>> EXPECTED_STEPS: Test1,test1,TEST1'
        + : '>>>>> Test1: 0'
        + : '>>>>> test1: 1'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "Test1": TestStatus.PASSED.value,
            "test1": TestStatus.FAILED.value,
            "TEST1": TestStatus.SKIPPED.value,
        }
        assert result == expected

    def test_duplicate_step_execution(self):
        """Test that if a step is executed multiple times, the last result wins."""
        log = """
        + : '>>>>> EXPECTED_STEPS: test1,test2'
        + : '>>>>> test1: 1'
        + : '>>>>> test2: 0'
        + : '>>>>> test1: 0'
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "test1": TestStatus.PASSED.value,  # Last execution wins
            "test2": TestStatus.PASSED.value,
        }
        assert result == expected

    def test_realistic_scenario(self):
        """Test a realistic multi-step test scenario."""
        log = """
        2025-10-03 12:23:12,866 - INFO - Starting comprehensive test suite
        + : '>>>>> EXPECTED_STEPS: environment_setup,dependency_check,static_analysis,unit_tests,integration_tests,performance_tests,security_scan,cleanup'
        
        Setting up test environment...
        + : '>>>>> environment_setup: 0'
        
        Checking dependencies...
        + : '>>>>> dependency_check: 0'
        
        Running static analysis...
        + : '>>>>> static_analysis: 1'
        Static analysis found issues, continuing with tests
        
        Running unit tests...
        + : '>>>>> unit_tests: 0'
        
        Running integration tests...
        + : '>>>>> integration_tests: 1'
        Integration tests failed, skipping performance tests and security scan
        
        Cleaning up...
        + : '>>>>> cleanup: 0'
        
        Test suite completed with failures
        """
        
        result = parse_log_generic_exit_codes(log, None)
        
        expected = {
            "environment_setup": TestStatus.PASSED.value,
            "dependency_check": TestStatus.PASSED.value,
            "static_analysis": TestStatus.FAILED.value,
            "unit_tests": TestStatus.PASSED.value,
            "integration_tests": TestStatus.FAILED.value,
            "performance_tests": TestStatus.SKIPPED.value,  # Not executed
            "security_scan": TestStatus.SKIPPED.value,      # Not executed
            "cleanup": TestStatus.PASSED.value,
        }
        assert result == expected
