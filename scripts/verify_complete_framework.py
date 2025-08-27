#!/usr/bin/env python3
"""
Complete Framework Verification

This script verifies that all components of the 6-phase BDD100K evaluation framework
are working correctly and generates a final comprehensive status report.
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class FrameworkVerifier:
    """Comprehensive verification of the BDD100K evaluation framework."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results = {
            'verification_timestamp': self.timestamp,
            'phase_results': {},
            'component_tests': {},
            'integration_tests': {},
            'performance_benchmarks': {},
            'overall_status': 'unknown'
        }
        
        # Define all framework components
        self.components = {
            'Phase 1 - Quantitative Metrics': {
                'scripts': ['scripts/run_comprehensive_evaluation.py'],
                'modules': [
                    'src/evaluation/metrics/coco_metrics.py',
                    'src/evaluation/metrics/safety_metrics.py',
                    'src/evaluation/metrics/contextual_metrics.py'
                ],
                'expected_outputs': ['evaluation_results/evaluation_results.json']
            },
            'Phase 2 - Visualization Tools': {
                'scripts': ['scripts/generate_evaluation_visualizations.py', 'scripts/test_visualizations.py'],
                'modules': ['src/evaluation/visualization/detection_viz.py'],
                'expected_outputs': ['evaluation_results/visualizations/']
            },
            'Phase 3 - Failure Analysis': {
                'scripts': ['scripts/test_failure_analysis.py'],
                'modules': [
                    'src/evaluation/analysis/failure_analyzer.py',
                    'src/evaluation/analysis/pattern_detector.py'
                ],
                'expected_outputs': ['evaluation_results/failure_analysis_tests/comprehensive_failure_analysis_results.json']
            },
            'Phase 4 - Clustering Analysis': {
                'scripts': ['scripts/run_phase4_analysis.py'],
                'modules': ['src/evaluation/analysis/pattern_detector.py'],
                'expected_outputs': ['evaluation_results/phase4_clustering/phase4_clustering_results.json']
            },
            'Phase 5 - Comprehensive Reporting': {
                'scripts': ['scripts/generate_comprehensive_report.py'],
                'modules': [],
                'expected_outputs': ['evaluation_results/comprehensive_reports/']
            },
            'Phase 6 - Improvement Recommendations': {
                'scripts': ['scripts/generate_improvement_recommendations.py'],
                'modules': [],
                'expected_outputs': ['evaluation_results/improvement_recommendations/']
            }
        }
    
    def run_complete_verification(self) -> Dict:
        """Run complete framework verification."""
        
        print("🔍 BDD100K Evaluation Framework - Complete Verification")
        print("=" * 70)
        print(f"Verification started at: {self.timestamp}")
        print("")
        
        # Step 1: Verify file structure
        print("Step 1: Verifying file structure and components...")
        structure_results = self._verify_file_structure()
        self.results['component_tests']['file_structure'] = structure_results
        
        # Step 2: Test individual components
        print("\\nStep 2: Testing individual components...")
        component_results = self._test_individual_components()
        self.results['component_tests']['individual_tests'] = component_results
        
        # Step 3: Run integration tests
        print("\\nStep 3: Running integration tests...")
        integration_results = self._run_integration_tests()
        self.results['integration_tests'] = integration_results
        
        # Step 4: Verify phase outputs
        print("\\nStep 4: Verifying phase outputs...")
        output_results = self._verify_phase_outputs()
        self.results['phase_results'] = output_results
        
        # Step 5: Performance benchmarking
        print("\\nStep 5: Running performance benchmarks...")
        performance_results = self._run_performance_benchmarks()
        self.results['performance_benchmarks'] = performance_results
        
        # Step 6: Generate final assessment
        print("\\nStep 6: Generating final assessment...")
        final_assessment = self._generate_final_assessment()
        self.results['final_assessment'] = final_assessment
        
        # Save verification results
        self._save_verification_results()
        
        # Display summary
        self._display_verification_summary()
        
        return self.results
    
    def _verify_file_structure(self) -> Dict:
        """Verify that all required files and directories exist."""
        
        structure_results = {
            'missing_files': [],
            'present_files': [],
            'missing_directories': [],
            'present_directories': []
        }
        
        # Check all component files
        for phase, component in self.components.items():
            print(f"  Checking {phase}...")
            
            # Check scripts
            for script in component['scripts']:
                if Path(script).exists():
                    structure_results['present_files'].append(script)
                    print(f"    ✅ {script}")
                else:
                    structure_results['missing_files'].append(script)
                    print(f"    ❌ {script}")
            
            # Check modules
            for module in component['modules']:
                if Path(module).exists():
                    structure_results['present_files'].append(module)
                    print(f"    ✅ {module}")
                else:
                    structure_results['missing_files'].append(module)
                    print(f"    ❌ {module}")
        
        # Check critical directories
        critical_dirs = [
            'src/evaluation/metrics',
            'src/evaluation/visualization', 
            'src/evaluation/analysis',
            'evaluation_results'
        ]
        
        for dir_path in critical_dirs:
            if Path(dir_path).exists():
                structure_results['present_directories'].append(dir_path)
                print(f"    ✅ {dir_path}/")
            else:
                structure_results['missing_directories'].append(dir_path)
                print(f"    ❌ {dir_path}/")
        
        structure_results['structure_score'] = len(structure_results['present_files']) / max(
            len(structure_results['present_files']) + len(structure_results['missing_files']), 1
        )
        
        return structure_results
    
    def _test_individual_components(self) -> Dict:
        """Test individual components for basic functionality."""
        
        component_results = {}
        
        # Test Phase 3 components (they don't require model)
        print("  Testing Phase 3 - Failure Analysis...")
        try:
            result = subprocess.run(['python3', 'scripts/test_failure_analysis.py'], 
                                   capture_output=True, text=True, timeout=120)
            component_results['phase3_failure_analysis'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'execution_time': 'completed',
                'output_length': len(result.stdout) + len(result.stderr)
            }
            print(f"    ✅ Phase 3 test completed (return code: {result.returncode})")
        except Exception as e:
            component_results['phase3_failure_analysis'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"    ❌ Phase 3 test failed: {e}")
        
        # Test Phase 2 components
        print("  Testing Phase 2 - Visualization Tools...")
        try:
            result = subprocess.run(['python3', 'scripts/test_visualizations.py'], 
                                   capture_output=True, text=True, timeout=60)
            component_results['phase2_visualization'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'execution_time': 'completed',
                'output_length': len(result.stdout) + len(result.stderr)
            }
            print(f"    ✅ Phase 2 test completed (return code: {result.returncode})")
        except Exception as e:
            component_results['phase2_visualization'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"    ❌ Phase 2 test failed: {e}")
        
        # Test import capabilities for all modules
        print("  Testing module imports...")
        import_tests = {}
        
        try:
            from src.evaluation.metrics.coco_metrics import COCOEvaluator
            import_tests['coco_metrics'] = 'success'
            print("    ✅ COCOEvaluator import successful")
        except Exception as e:
            import_tests['coco_metrics'] = f'failed: {e}'
            print(f"    ❌ COCOEvaluator import failed: {e}")
        
        try:
            from src.evaluation.metrics.safety_metrics import SafetyMetrics
            import_tests['safety_metrics'] = 'success'
            print("    ✅ SafetyMetrics import successful")
        except Exception as e:
            import_tests['safety_metrics'] = f'failed: {e}'
            print(f"    ❌ SafetyMetrics import failed: {e}")
        
        try:
            from src.evaluation.visualization.detection_viz import DetectionVisualizer
            import_tests['detection_visualizer'] = 'success'
            print("    ✅ DetectionVisualizer import successful")
        except Exception as e:
            import_tests['detection_visualizer'] = f'failed: {e}'
            print(f"    ❌ DetectionVisualizer import failed: {e}")
        
        try:
            from src.evaluation.analysis.failure_analyzer import FailureAnalyzer
            import_tests['failure_analyzer'] = 'success'
            print("    ✅ FailureAnalyzer import successful")
        except Exception as e:
            import_tests['failure_analyzer'] = f'failed: {e}'
            print(f"    ❌ FailureAnalyzer import failed: {e}")
        
        try:
            from src.evaluation.analysis.pattern_detector import PerformancePatternDetector
            import_tests['pattern_detector'] = 'success'
            print("    ✅ PerformancePatternDetector import successful")
        except Exception as e:
            import_tests['pattern_detector'] = f'failed: {e}'
            print(f"    ❌ PerformancePatternDetector import failed: {e}")
        
        component_results['import_tests'] = import_tests
        
        return component_results
    
    def _run_integration_tests(self) -> Dict:
        """Run integration tests between components."""
        
        integration_results = {}
        
        # Test Phase 3 -> Phase 4 integration
        print("  Testing Phase 3 -> Phase 4 integration...")
        phase3_output = Path('evaluation_results/failure_analysis_tests/comprehensive_failure_analysis_results.json')
        
        if phase3_output.exists():
            try:
                # Run Phase 4 using Phase 3 output
                result = subprocess.run(['python3', 'scripts/run_phase4_analysis.py'], 
                                       capture_output=True, text=True, timeout=180)
                integration_results['phase3_to_phase4'] = {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'phase3_output_exists': True,
                    'phase4_completion': result.returncode == 0
                }
                print(f"    ✅ Phase 3->4 integration test completed (return code: {result.returncode})")
            except Exception as e:
                integration_results['phase3_to_phase4'] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"    ❌ Phase 3->4 integration failed: {e}")
        else:
            integration_results['phase3_to_phase4'] = {
                'status': 'skipped',
                'reason': 'Phase 3 output not found'
            }
            print("    ⚠️ Phase 3 output not found, skipping integration test")
        
        # Test Phase 4 -> Phase 5 integration
        print("  Testing Phase 4 -> Phase 5 integration...")
        phase4_output = Path('evaluation_results/phase4_clustering/phase4_clustering_results.json')
        
        if phase4_output.exists():
            try:
                # Run Phase 5 using Phase 4 output
                result = subprocess.run(['python3', 'scripts/generate_comprehensive_report.py'], 
                                       capture_output=True, text=True, timeout=120)
                integration_results['phase4_to_phase5'] = {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'phase4_output_exists': True,
                    'phase5_completion': result.returncode == 0
                }
                print(f"    ✅ Phase 4->5 integration test completed (return code: {result.returncode})")
            except Exception as e:
                integration_results['phase4_to_phase5'] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"    ❌ Phase 4->5 integration failed: {e}")
        else:
            integration_results['phase4_to_phase5'] = {
                'status': 'skipped',
                'reason': 'Phase 4 output not found'
            }
            print("    ⚠️ Phase 4 output not found, skipping integration test")
        
        # Test Phase 5 -> Phase 6 integration
        print("  Testing Phase 5 -> Phase 6 integration...")
        phase5_output = Path('evaluation_results/comprehensive_reports')
        
        if phase5_output.exists():
            try:
                # Run Phase 6
                result = subprocess.run(['python3', 'scripts/generate_improvement_recommendations.py'], 
                                       capture_output=True, text=True, timeout=120)
                integration_results['phase5_to_phase6'] = {
                    'status': 'passed' if result.returncode == 0 else 'failed',
                    'phase5_output_exists': True,
                    'phase6_completion': result.returncode == 0
                }
                print(f"    ✅ Phase 5->6 integration test completed (return code: {result.returncode})")
            except Exception as e:
                integration_results['phase5_to_phase6'] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"    ❌ Phase 5->6 integration failed: {e}")
        else:
            integration_results['phase5_to_phase6'] = {
                'status': 'skipped',
                'reason': 'Phase 5 output not found'
            }
            print("    ⚠️ Phase 5 output not found, skipping integration test")
        
        return integration_results
    
    def _verify_phase_outputs(self) -> Dict:
        """Verify that all phase outputs are present and valid."""
        
        output_results = {}
        
        # Define expected outputs for each phase
        expected_outputs = {
            'Phase 1': [
                'evaluation_results/evaluation_results.json'
            ],
            'Phase 2': [
                'evaluation_results/visualizations/',
                'evaluation_results/visualization_tests/'
            ],
            'Phase 3': [
                'evaluation_results/failure_analysis_tests/comprehensive_failure_analysis_results.json'
            ],
            'Phase 4': [
                'evaluation_results/phase4_clustering/phase4_clustering_results.json'
            ],
            'Phase 5': [
                'evaluation_results/comprehensive_reports/executive_summary_report.md',
                'evaluation_results/comprehensive_reports/technical_evaluation_report.md',
                'evaluation_results/comprehensive_reports/model_improvement_action_plan.md',
                'evaluation_results/comprehensive_reports/comprehensive_evaluation_dashboard.png'
            ],
            'Phase 6': [
                'evaluation_results/improvement_recommendations/executive_improvement_recommendations.md',
                'evaluation_results/improvement_recommendations/technical_implementation_guide.md',
                'evaluation_results/improvement_recommendations/data_improvement_strategy.md',
                'evaluation_results/improvement_recommendations/safety_enhancement_plan.md',
                'evaluation_results/improvement_recommendations/comprehensive_improvement_recommendations.json'
            ]
        }
        
        for phase, outputs in expected_outputs.items():
            print(f"  Verifying {phase} outputs...")
            
            phase_result = {
                'total_expected': len(outputs),
                'present': 0,
                'missing': [],
                'present_files': []
            }
            
            for output_path in outputs:
                path = Path(output_path)
                
                if path.exists():
                    phase_result['present'] += 1
                    phase_result['present_files'].append(output_path)
                    print(f"    ✅ {output_path}")
                    
                    # Additional validation for JSON files
                    if output_path.endswith('.json'):
                        try:
                            with open(path, 'r') as f:
                                data = json.load(f)
                            print(f"      📄 Valid JSON with {len(data)} top-level keys")
                        except Exception as e:
                            print(f"      ⚠️ JSON validation failed: {e}")
                else:
                    phase_result['missing'].append(output_path)
                    print(f"    ❌ {output_path}")
            
            phase_result['completion_rate'] = phase_result['present'] / phase_result['total_expected']
            output_results[phase] = phase_result
        
        return output_results
    
    def _run_performance_benchmarks(self) -> Dict:
        """Run basic performance benchmarks."""
        
        benchmark_results = {}
        
        print("  Running component performance benchmarks...")
        
        # Benchmark visualization generation
        print("    Benchmarking visualization generation...")
        try:
            start_time = time.time()
            
            # Quick visualization test
            from src.evaluation.visualization.detection_viz import DetectionVisualizer
            visualizer = DetectionVisualizer()
            
            # Create a simple test
            import numpy as np
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_gt = [{'category_id': 0, 'bbox': [100, 100, 50, 50]}]
            test_preds = [{'category_id': 0, 'bbox': [105, 105, 45, 45], 'score': 0.8}]
            
            _ = visualizer.visualize_detections(test_image, test_gt, test_preds)
            
            end_time = time.time()
            
            benchmark_results['visualization_generation'] = {
                'execution_time_seconds': round(end_time - start_time, 3),
                'status': 'success'
            }
            print(f"      ✅ Visualization benchmark: {end_time - start_time:.3f}s")
            
        except Exception as e:
            benchmark_results['visualization_generation'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"      ❌ Visualization benchmark failed: {e}")
        
        # Benchmark failure analysis
        print("    Benchmarking failure analysis...")
        try:
            start_time = time.time()
            
            from src.evaluation.analysis.failure_analyzer import FailureAnalyzer
            analyzer = FailureAnalyzer()
            
            # Create sample data
            sample_gt = [
                {'category_id': 0, 'bbox': [100, 100, 50, 50], 'image_id': 'test_001'},
                {'category_id': 1, 'bbox': [200, 200, 60, 60], 'image_id': 'test_001'}
            ]
            sample_preds = [
                {'category_id': 0, 'bbox': [105, 105, 45, 45], 'score': 0.8, 'image_id': 'test_001'},
                {'category_id': 2, 'bbox': [300, 300, 40, 40], 'score': 0.6, 'image_id': 'test_001'}
            ]
            sample_metadata = [{'image_id': 'test_001', 'weather': 'clear', 'lighting': 'day'}]
            
            _ = analyzer.analyze_failures(sample_preds, sample_gt, sample_metadata)
            
            end_time = time.time()
            
            benchmark_results['failure_analysis'] = {
                'execution_time_seconds': round(end_time - start_time, 3),
                'status': 'success'
            }
            print(f"      ✅ Failure analysis benchmark: {end_time - start_time:.3f}s")
            
        except Exception as e:
            benchmark_results['failure_analysis'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"      ❌ Failure analysis benchmark failed: {e}")
        
        # Benchmark pattern detection
        print("    Benchmarking pattern detection...")
        try:
            start_time = time.time()
            
            from src.evaluation.analysis.pattern_detector import PerformancePatternDetector
            detector = PerformancePatternDetector()
            
            # Create sample evaluation results
            sample_results = {
                'coco_metrics': {
                    'mAP': 0.45,
                    'per_class_AP': {
                        'pedestrian': 0.4, 'car': 0.6, 'bicycle': 0.3
                    }
                },
                'safety_metrics': {
                    'per_class_safety': {
                        'pedestrian': {'recall': 0.7, 'precision': 0.6}
                    }
                },
                'contextual_metrics': {
                    'weather_performance': {
                        'clear': {'mean_ap': 0.5},
                        'rain': {'mean_ap': 0.4}
                    }
                }
            }
            
            _ = detector.detect_performance_patterns(sample_results)
            
            end_time = time.time()
            
            benchmark_results['pattern_detection'] = {
                'execution_time_seconds': round(end_time - start_time, 3),
                'status': 'success'
            }
            print(f"      ✅ Pattern detection benchmark: {end_time - start_time:.3f}s")
            
        except Exception as e:
            benchmark_results['pattern_detection'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"      ❌ Pattern detection benchmark failed: {e}")
        
        return benchmark_results
    
    def _generate_final_assessment(self) -> Dict:
        """Generate final assessment of the framework."""
        
        assessment = {
            'overall_status': 'unknown',
            'readiness_level': 'unknown',
            'critical_issues': [],
            'recommendations': [],
            'framework_score': 0.0
        }
        
        # Analyze results
        structure_score = self.results['component_tests']['file_structure'].get('structure_score', 0)
        
        # Count successful components
        component_tests = self.results['component_tests']['individual_tests']
        import_successes = sum(1 for status in component_tests.get('import_tests', {}).values() 
                              if status == 'success')
        total_imports = len(component_tests.get('import_tests', {}))
        
        # Count successful integrations
        integration_tests = self.results['integration_tests']
        integration_successes = sum(1 for test_result in integration_tests.values() 
                                   if test_result.get('status') == 'passed')
        total_integrations = len(integration_tests)
        
        # Count phase output completions
        phase_results = self.results['phase_results']
        total_completion_rate = np.mean([
            phase_data.get('completion_rate', 0) 
            for phase_data in phase_results.values()
        ]) if phase_results else 0
        
        # Calculate overall framework score
        scores = [
            structure_score * 0.2,  # 20% for file structure
            (import_successes / max(total_imports, 1)) * 0.3,  # 30% for component functionality
            (integration_successes / max(total_integrations, 1)) * 0.3,  # 30% for integration
            total_completion_rate * 0.2  # 20% for phase outputs
        ]
        
        framework_score = sum(scores)
        assessment['framework_score'] = round(framework_score, 3)
        
        # Determine overall status
        if framework_score >= 0.9:
            assessment['overall_status'] = 'excellent'
            assessment['readiness_level'] = 'production_ready'
        elif framework_score >= 0.8:
            assessment['overall_status'] = 'good'
            assessment['readiness_level'] = 'ready_with_minor_fixes'
        elif framework_score >= 0.7:
            assessment['overall_status'] = 'acceptable'
            assessment['readiness_level'] = 'needs_improvement'
        else:
            assessment['overall_status'] = 'needs_work'
            assessment['readiness_level'] = 'major_improvements_needed'
        
        # Identify critical issues
        if structure_score < 0.8:
            assessment['critical_issues'].append('Missing critical framework files')
        
        if import_successes < total_imports:
            assessment['critical_issues'].append('Component import failures detected')
        
        if integration_successes == 0:
            assessment['critical_issues'].append('No successful integration tests')
        
        if total_completion_rate < 0.5:
            assessment['critical_issues'].append('Many phase outputs missing')
        
        # Generate recommendations
        if framework_score < 0.8:
            assessment['recommendations'].extend([
                'Run individual phase scripts to generate missing outputs',
                'Fix import errors in failing components',
                'Verify all required dependencies are installed'
            ])
        
        if len(assessment['critical_issues']) == 0:
            assessment['recommendations'].append('Framework is ready for production use')
        
        return assessment
    
    def _save_verification_results(self):
        """Save verification results to file."""
        
        results_dir = Path('evaluation_results/framework_verification')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f'framework_verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\\n📊 Verification results saved to: {results_file}")
    
    def _display_verification_summary(self):
        """Display comprehensive verification summary."""
        
        print("\\n" + "=" * 70)
        print("🔍 BDD100K EVALUATION FRAMEWORK - VERIFICATION SUMMARY")
        print("=" * 70)
        
        # Overall assessment
        assessment = self.results['final_assessment']
        
        status_emoji = {
            'excellent': '🟢',
            'good': '🟡', 
            'acceptable': '🟠',
            'needs_work': '🔴'
        }
        
        readiness_emoji = {
            'production_ready': '✅',
            'ready_with_minor_fixes': '⚠️',
            'needs_improvement': '⚠️',
            'major_improvements_needed': '❌'
        }
        
        print(f"\\n📊 **OVERALL STATUS**: {status_emoji.get(assessment['overall_status'], '❓')} {assessment['overall_status'].upper()}")
        print(f"🎯 **READINESS LEVEL**: {readiness_emoji.get(assessment['readiness_level'], '❓')} {assessment['readiness_level'].replace('_', ' ').upper()}")
        print(f"📈 **FRAMEWORK SCORE**: {assessment['framework_score']:.1%}")
        print("")
        
        # Component breakdown
        print("📋 **COMPONENT STATUS BREAKDOWN**:")
        
        structure_score = self.results['component_tests']['file_structure']['structure_score']
        print(f"  - File Structure: {structure_score:.1%}")
        
        component_tests = self.results['component_tests']['individual_tests']
        import_tests = component_tests.get('import_tests', {})
        import_success_rate = sum(1 for status in import_tests.values() if status == 'success') / max(len(import_tests), 1)
        print(f"  - Component Imports: {import_success_rate:.1%}")
        
        integration_tests = self.results['integration_tests']
        integration_success_rate = sum(1 for test in integration_tests.values() if test.get('status') == 'passed') / max(len(integration_tests), 1)
        print(f"  - Integration Tests: {integration_success_rate:.1%}")
        
        phase_results = self.results['phase_results']
        if phase_results:
            avg_completion = np.mean([phase.get('completion_rate', 0) for phase in phase_results.values()])
            print(f"  - Phase Outputs: {avg_completion:.1%}")
        
        # Phase-by-phase status
        print("\\n🎯 **PHASE-BY-PHASE STATUS**:")
        for phase, results in phase_results.items():
            completion_rate = results.get('completion_rate', 0)
            status_symbol = '✅' if completion_rate >= 0.8 else '⚠️' if completion_rate >= 0.5 else '❌'
            print(f"  {status_symbol} {phase}: {completion_rate:.1%} ({results['present']}/{results['total_expected']} outputs)")
        
        # Critical issues
        if assessment['critical_issues']:
            print("\\n🚨 **CRITICAL ISSUES**:")
            for issue in assessment['critical_issues']:
                print(f"  - {issue}")
        
        # Recommendations
        print("\\n💡 **RECOMMENDATIONS**:")
        for rec in assessment['recommendations']:
            print(f"  - {rec}")
        
        # Performance benchmarks
        benchmarks = self.results.get('performance_benchmarks', {})
        if benchmarks:
            print("\\n⚡ **PERFORMANCE BENCHMARKS**:")
            for component, result in benchmarks.items():
                if result.get('status') == 'success':
                    time_taken = result.get('execution_time_seconds', 0)
                    print(f"  - {component.replace('_', ' ').title()}: {time_taken:.3f}s")
        
        # Final conclusion
        print("\\n🎬 **CONCLUSION**:")
        
        if assessment['overall_status'] in ['excellent', 'good']:
            print("✅ The BDD100K evaluation framework is working correctly and ready for use.")
            print("   All major components are functional with comprehensive analysis capabilities.")
        elif assessment['overall_status'] == 'acceptable':
            print("⚠️ The framework is mostly functional but has some areas that need attention.")
            print("   Address the identified issues to achieve full readiness.")
        else:
            print("❌ The framework requires significant work before it can be used reliably.")
            print("   Focus on resolving critical issues first.")
        
        print("\\n" + "=" * 70)
        print(f"Verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


def main():
    """Run complete framework verification."""
    
    verifier = FrameworkVerifier()
    
    try:
        results = verifier.run_complete_verification()
        
        # Return appropriate exit code
        final_assessment = results.get('final_assessment', {})
        framework_score = final_assessment.get('framework_score', 0)
        
        if framework_score >= 0.8:
            return 0  # Success
        elif framework_score >= 0.6:
            return 1  # Warning
        else:
            return 2  # Error
            
    except Exception as e:
        print(f"\\n❌ Framework verification failed with error: {e}")
        return 3  # Critical error


if __name__ == "__main__":
    import sys
    sys.exit(main())