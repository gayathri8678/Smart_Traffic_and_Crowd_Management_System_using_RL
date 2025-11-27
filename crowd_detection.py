"""
Crowd Detection System - Input Video Processing
Analyzes crowd videos using trained RL model
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
import os
from datetime import datetime
import json

# ==================== LOAD MODEL ARCHITECTURE ====================

class CNN_FeatureExtractor(nn.Module):
    def __init__(self):
        super(CNN_FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        
    def forward(self, x):
        return self.conv(x)


class DQN_Network(nn.Module):
    def __init__(self, n_actions):
        super(DQN_Network, self).__init__()
        self.feature_extractor = CNN_FeatureExtractor()
        test_input = torch.zeros(1, 3, 224, 224)
        conv_output = self.feature_extractor(test_input)
        self.feature_size = conv_output.view(1, -1).size(1)
        
        self.decision_network = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_actions)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return self.decision_network(features)


# ==================== PERSON DETECTOR ====================

class PersonDetector:
    """Detects people in crowd videos"""
    
    def __init__(self):
        # HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Background subtraction for motion
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # Try to load upper body cascade
        self.body_cascade = None
        cascade_path = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        if os.path.exists(cascade_path):
            self.body_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect(self, frame):
        """Detect people in frame"""
        people = []
        
        # Method 1: HOG detection (more accurate but slower)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster detection
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        boxes, weights = self.hog.detectMultiScale(
            small_frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        for (x, y, w, h), weight in zip(boxes, weights):
            if weight > 0.5:  # Confidence threshold
                # Scale back to original size
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                people.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': w * h,
                    'type': 'person',
                    'confidence': float(weight)
                })
        
        # Method 2: Motion-based detection
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask[fg_mask == 127] = 0  # Remove shadows
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Person-sized objects
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / float(w) if w > 0 else 0
                
                # Person-like aspect ratio (taller than wide)
                if 1.5 < aspect_ratio < 4.0:
                    # Check if not already detected
                    is_duplicate = False
                    for person in people:
                        px, py, pw, ph = person['bbox']
                        if abs(x - px) < 50 and abs(y - py) < 50:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        people.append({
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2),
                            'area': area,
                            'type': 'person',
                            'confidence': 0.7
                        })
        
        return people


# ==================== CROWD ANALYZER ====================

class CrowdAnalyzer:
    """Analyzes crowd density and flow patterns"""
    
    def __init__(self):
        self.crowd_history = deque(maxlen=30)
        self.density_zones = None
    
    def analyze(self, people, frame_shape):
        """Analyze crowd situation"""
        
        # Store history
        self.crowd_history.append(people)
        
        # Calculate crowd density
        total_area = sum(p['area'] for p in people)
        frame_area = frame_shape[0] * frame_shape[1]
        density = min(total_area / frame_area, 1.0)
        
        # Calculate density zones (grid-based)
        density_map = self._calculate_density_map(people, frame_shape)
        
        # Calculate crowd flow
        flow_speed, flow_direction = self._estimate_flow()
        
        # Calculate congestion points
        congestion_risk = self._calculate_congestion_risk(people, density_map)
        
        # Calculate stampede risk (based on density + flow)
        stampede_risk = self._calculate_stampede_risk(density, flow_speed, congestion_risk)
        
        return {
            'person_count': len(people),
            'density': density,
            'density_map': density_map,
            'flow_speed': flow_speed,
            'flow_direction': flow_direction,
            'congestion_risk': congestion_risk,
            'stampede_risk': stampede_risk
        }
    
    def _calculate_density_map(self, people, frame_shape):
        """Create density heatmap"""
        grid_size = 8
        h, w = frame_shape[:2]
        cell_h, cell_w = h // grid_size, w // grid_size
        
        density_map = np.zeros((grid_size, grid_size))
        
        for person in people:
            cx, cy = person['center']
            grid_x = min(int(cx / cell_w), grid_size - 1)
            grid_y = min(int(cy / cell_h), grid_size - 1)
            density_map[grid_y, grid_x] += 1
        
        # Normalize
        if density_map.max() > 0:
            density_map = density_map / density_map.max()
        
        return density_map
    
    def _estimate_flow(self):
        """Estimate crowd flow using optical flow"""
        if len(self.crowd_history) < 2:
            return 0.0, 0.0
        
        prev_people = self.crowd_history[-2]
        curr_people = self.crowd_history[-1]
        
        movements = []
        
        for curr_p in curr_people:
            for prev_p in prev_people:
                dx = curr_p['center'][0] - prev_p['center'][0]
                dy = curr_p['center'][1] - prev_p['center'][1]
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < 30:  # Same person
                    movements.append((dx, dy))
                    break
        
        if not movements:
            return 0.0, 0.0
        
        avg_dx = np.mean([m[0] for m in movements])
        avg_dy = np.mean([m[1] for m in movements])
        
        speed = np.sqrt(avg_dx**2 + avg_dy**2)
        direction = np.arctan2(avg_dy, avg_dx)
        
        return float(speed), float(direction)
    
    def _calculate_congestion_risk(self, people, density_map):
        """Calculate congestion risk in high-density zones"""
        if len(people) < 5:
            return 0.0
        
        # Find high-density zones
        high_density_zones = (density_map > 0.7).sum()
        
        # Calculate proximity risk
        risk = 0.0
        for i, p1 in enumerate(people):
            close_neighbors = 0
            for p2 in people[i+1:]:
                dx = p1['center'][0] - p2['center'][0]
                dy = p1['center'][1] - p2['center'][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 50:  # Very close
                    close_neighbors += 1
            
            if close_neighbors > 3:  # Crowded
                risk += 0.1
        
        congestion = min((high_density_zones / 64.0) * 0.5 + min(risk, 0.5), 1.0)
        return congestion
    
    def _calculate_stampede_risk(self, density, flow_speed, congestion):
        """Calculate stampede risk"""
        # High density + high speed + congestion = stampede risk
        risk = (density * 0.4 + 
                min(flow_speed / 10.0, 1.0) * 0.3 + 
                congestion * 0.3)
        
        return min(risk, 1.0)


# ==================== CROWD DETECTION SYSTEM ====================

class CrowdDetectionSystem:
    """Complete Crowd Detection & Management System"""
    
    def __init__(self, model_path='models/trained_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load RL model
        self.model = DQN_Network(n_actions=5).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Initialize detectors
        self.person_detector = PersonDetector()
        self.crowd_analyzer = CrowdAnalyzer()
        
        # Action labels for crowd management
        self.action_labels = [
            'SAFE - Normal Crowd Flow',
            'WARNING - Monitor Density',
            'ALERT - Control Entry',
            'CRITICAL - Restrict Access',
            'EMERGENCY - Evacuate Area'
        ]
        
        self.action_colors = [
            (0, 255, 0),      # Green
            (0, 255, 255),    # Yellow
            (0, 165, 255),    # Orange
            (0, 100, 255),    # Dark Orange
            (0, 0, 255)       # Red
        ]
    
    def process_video(self, input_path, output_path=None, display=True):
        """Process crowd video"""
        
        print("\n" + "="*60)
        print("CROWD DETECTION SYSTEM - VIDEO PROCESSING")
        print("="*60)
        print(f"Input: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open video file")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ Resolution: {width}x{height}")
        print(f"✓ FPS: {fps}")
        print(f"✓ Total Frames: {total_frames}")
        
        # Setup output
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 250))
            print(f"✓ Output: {output_path}")
        else:
            out = None
        
        print("\nProcessing...")
        
        # Statistics
        stats = {
            'frames_processed': 0,
            'people_detected': [],
            'actions_taken': [0] * 5,
            'max_density': 0,
            'max_stampede_risk': 0,
            'critical_moments': []
        }
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect people
            people = self.person_detector.detect(frame)
            
            # Analyze crowd
            analysis = self.crowd_analyzer.analyze(people, frame.shape)
            
            # Get RL decision
            processed_frame = cv2.resize(frame, (224, 224))
            with torch.no_grad():
                state = torch.FloatTensor(processed_frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
                state = state / 255.0
                q_values = self.model(state)
                action = q_values.argmax().item()
            
            # Update statistics
            stats['frames_processed'] += 1
            stats['people_detected'].append(analysis['person_count'])
            stats['actions_taken'][action] += 1
            stats['max_density'] = max(stats['max_density'], analysis['density'])
            stats['max_stampede_risk'] = max(stats['max_stampede_risk'], analysis['stampede_risk'])
            
            # Record critical moments
            if action >= 3:  # Critical or Emergency
                stats['critical_moments'].append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'action': self.action_labels[action],
                    'people': analysis['person_count'],
                    'density': analysis['density'],
                    'risk': analysis['stampede_risk']
                })
            
            # Visualize
            output_frame = self._visualize(frame, people, analysis, action)
            
            # Display
            if display:
                cv2.imshow('Crowd Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write output
            if out:
                out.write(output_frame)
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% | People: {analysis['person_count']} | "
                      f"Density: {analysis['density']*100:.0f}% | Action: {self.action_labels[action]}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print summary
        self._print_summary(stats)
        
        # Save report
        self._save_report(input_path, stats)
    
    def _visualize(self, frame, people, analysis, action):
        """Add visualization overlays"""
        output = frame.copy()
        
        # Draw person bounding boxes
        for person in people:
            x, y, w, h = person['bbox']
            color = (0, 255, 0) if person['confidence'] > 0.7 else (0, 255, 255)
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
            
            # Draw confidence
            conf_text = f"{person['confidence']:.2f}"
            cv2.putText(output, conf_text, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw density heatmap overlay
        density_map = analysis['density_map']
        heatmap = self._create_heatmap(density_map, frame.shape)
        output = cv2.addWeighted(output, 0.7, heatmap, 0.3, 0)
        
        # Draw info panel
        panel_height = 250
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Action status
        action_color = self.action_colors[action]
        cv2.rectangle(panel, (0, 0), (frame.shape[1], 50), action_color, -1)
        cv2.putText(panel, self.action_labels[action], (20, 35),
                   cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 3)
        
        # Statistics
        y_offset = 70
        stats_text = [
            f"People Detected: {analysis['person_count']}",
            f"Crowd Density: {analysis['density']*100:.1f}%",
            f"Flow Speed: {analysis['flow_speed']:.1f} px/frame",
            f"Congestion Risk: {analysis['congestion_risk']*100:.1f}%",
            f"Stampede Risk: {analysis['stampede_risk']*100:.1f}%"
        ]
        
        for text in stats_text:
            cv2.putText(panel, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 35
        
        # Risk meter
        self._draw_risk_meter(panel, analysis['stampede_risk'], frame.shape[1] - 150, 70)
        
        # Combine
        output = np.vstack([output, panel])
        
        return output
    
    def _create_heatmap(self, density_map, frame_shape):
        """Create density heatmap overlay"""
        h, w = frame_shape[:2]
        heatmap = cv2.resize((density_map * 255).astype(np.uint8), (w, h))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap
    
    def _draw_risk_meter(self, panel, risk, x, y):
        """Draw risk meter"""
        meter_width = 120
        meter_height = 150
        
        # Border
        cv2.rectangle(panel, (x, y), (x+meter_width, y+meter_height), (255, 255, 255), 2)
        
        # Fill based on risk
        fill_height = int(risk * meter_height)
        if risk < 0.3:
            color = (0, 255, 0)
        elif risk < 0.6:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        
        cv2.rectangle(panel, 
                     (x+2, y+meter_height-fill_height), 
                     (x+meter_width-2, y+meter_height-2), 
                     color, -1)
        
        # Label
        cv2.putText(panel, "RISK", (x+30, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(panel, f"{risk*100:.0f}%", (x+35, y+meter_height+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _print_summary(self, stats):
        """Print processing summary"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"✓ Total Frames: {stats['frames_processed']}")
        print(f"✓ Avg People: {np.mean(stats['people_detected']):.1f}")
        print(f"✓ Peak Count: {max(stats['people_detected'])}")
        print(f"✓ Max Density: {stats['max_density']*100:.1f}%")
        print(f"✓ Max Stampede Risk: {stats['max_stampede_risk']*100:.1f}%")
        print(f"✓ Critical Moments: {len(stats['critical_moments'])}")
        
        print("\nActions Distribution:")
        for i, count in enumerate(stats['actions_taken']):
            percentage = (count / stats['frames_processed']) * 100
            print(f"  {self.action_labels[i]}: {count} ({percentage:.1f}%)")
        
        if stats['critical_moments']:
            print("\nCritical Moments:")
            for moment in stats['critical_moments'][:5]:  # Show first 5
                print(f"  Frame {moment['frame']} ({moment['time']:.1f}s): "
                      f"{moment['action']} - {moment['people']} people")
        
        print("="*60 + "\n")
    
    def _save_report(self, input_path, stats):
        """Save analysis report"""
        report_path = f"output_videos/crowd/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        report = {
            'input_file': input_path,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'frames_processed': stats['frames_processed'],
                'avg_people': float(np.mean(stats['people_detected'])),
                'peak_count': int(max(stats['people_detected'])),
                'max_density': float(stats['max_density']),
                'max_stampede_risk': float(stats['max_stampede_risk']),
                'critical_moments_count': len(stats['critical_moments']),
                'actions_distribution': {
                    self.action_labels[i]: int(count) 
                    for i, count in enumerate(stats['actions_taken'])
                }
            },
            'critical_moments': stats['critical_moments']
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"✓ Report saved: {report_path}")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n👥 CROWD DETECTION SYSTEM 👥\n")
    
    # Initialize system
    system = CrowdDetectionSystem(model_path='models/trained_model.pth')
    
    # Example usage
    input_video = 'input_videos/crowd/crowd_video.mp4'
    output_video = 'output_videos/crowd/processed_crowd.mp4'
    
    # Check if input exists
    if not os.path.exists(input_video):
        print(f"⚠️  Input video not found: {input_video}")
        print("Creating demo with webcam... Press 'q' to quit")
        input_video = 0  # Use webcam
        output_video = None
    
    # Process video
    system.process_video(
        input_path=input_video,
        output_path=output_video,
        display=True
    )
    
    print("\n✅ Crowd detection completed!\n")