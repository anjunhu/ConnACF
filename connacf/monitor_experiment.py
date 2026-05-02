#!/usr/bin/env python3
"""
Real-time Experiment Monitor for ConnaCF
Monitors ConnaCF experiment logs in real-time to help debug hanging issues.
"""

import os
import time
import json
import argparse
from datetime import datetime
from collections import defaultdict


class ExperimentLogMonitor:
    """Real-time monitor for ConnaCF experiment logs"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.last_positions = {}
        self.stats = defaultdict(int)
        self.last_activity = None
        
    def find_latest_experiment(self):
        """Find the most recent experiment logs"""
        if not os.path.exists(self.log_dir):
            return None
        
        experiments = []
        for filename in os.listdir(self.log_dir):
            if filename.endswith('_main.log'):
                filepath = os.path.join(self.log_dir, filename)
                mtime = os.path.getmtime(filepath)
                experiments.append((mtime, filename, filepath))
        
        if not experiments:
            return None
        
        # Return most recent experiment
        experiments.sort(reverse=True)
        return experiments[0][2]
    
    def monitor_main_log(self, log_file: str):
        """Monitor main log file for new entries"""
        if not os.path.exists(log_file):
            return []
        
        current_pos = self.last_positions.get(log_file, 0)
        new_lines = []
        
        try:
            with open(log_file, 'r') as f:
                f.seek(current_pos)
                new_lines = f.readlines()
                self.last_positions[log_file] = f.tell()
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        return new_lines
    
    def monitor_detailed_log(self, log_file: str):
        """Monitor detailed JSONL log for statistics"""
        detailed_file = log_file.replace('_main.log', '_detailed.jsonl')
        if not os.path.exists(detailed_file):
            return []
        
        current_pos = self.last_positions.get(detailed_file, 0)
        new_entries = []
        
        try:
            with open(detailed_file, 'r') as f:
                f.seek(current_pos)
                for line in f:
                    if line.strip():
                        try:
                            entry = json.loads(line.strip())
                            new_entries.append(entry)
                        except json.JSONDecodeError:
                            continue
                self.last_positions[detailed_file] = f.tell()
        except Exception as e:
            print(f"Error reading detailed log: {e}")
        
        return new_entries
    
    def update_stats(self, entries):
        """Update statistics from log entries"""
        for entry in entries:
            event_type = entry.get('event_type', 'unknown')
            self.stats[event_type] += 1
            
            if 'timestamp' in entry:
                self.last_activity = entry['timestamp']
    
    def print_status(self):
        """Print current status"""
        current_time = datetime.now().isoformat()
        print(f"\n[{current_time}] === EXPERIMENT STATUS ===")
        
        if self.last_activity:
            print(f"Last activity: {self.last_activity}")
            
            # Calculate time since last activity
            try:
                last_time = datetime.fromisoformat(self.last_activity.replace('Z', '+00:00'))
                current_dt = datetime.now()
                if last_time.tzinfo:
                    import pytz
                    current_dt = current_dt.replace(tzinfo=pytz.UTC)
                
                time_diff = (current_dt - last_time).total_seconds()
                print(f"Time since last activity: {time_diff:.1f} seconds")
                
                if time_diff > 120:
                    print("⚠️  WARNING: No activity for over 2 minutes - possible hang!")
                elif time_diff > 60:
                    print("⚠️  CAUTION: No activity for over 1 minute")
                
            except Exception:
                print("Could not calculate time difference")
        else:
            print("No activity detected yet")
        
        print("\nEvent Statistics:")
        for event_type, count in sorted(self.stats.items()):
            print(f"  {event_type}: {count}")
        
        # Calculate success rates
        total_interactions = self.stats.get('interaction_start', 0)
        successful = self.stats.get('interaction_success', 0)
        failed = self.stats.get('interaction_failure', 0)
        
        if total_interactions > 0:
            success_rate = successful / total_interactions * 100
            print(f"\nInteraction Success Rate: {successful}/{total_interactions} ({success_rate:.1f}%)")
            
            if failed > 0:
                print(f"Failed interactions: {failed}")
        
        print("=" * 50)
    
    def monitor_experiment(self, log_file: str = None, update_interval: float = 5.0):
        """Monitor experiment in real-time"""
        if log_file is None:
            log_file = self.find_latest_experiment()
            if log_file is None:
                print(f"No experiment logs found in {self.log_dir}")
                return
        
        print(f"Monitoring experiment log: {log_file}")
        print(f"Update interval: {update_interval} seconds")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                # Monitor main log for new lines
                new_lines = self.monitor_main_log(log_file)
                for line in new_lines:
                    print(line.rstrip())
                
                # Monitor detailed log for statistics
                new_entries = self.monitor_detailed_log(log_file)
                self.update_stats(new_entries)
                
                # Print status every few updates
                if len(new_lines) > 0 or len(new_entries) > 0:
                    self.print_status()
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.print_status()


def main():
    parser = argparse.ArgumentParser(description="Monitor ConnaCF experiment logs in real-time")
    parser.add_argument("--log_dir", "-d", type=str, default="llm_interaction_logs", 
                       help="Directory containing experiment logs")
    parser.add_argument("--log_file", "-f", type=str, help="Specific log file to monitor")
    parser.add_argument("--interval", "-i", type=float, default=5.0, 
                       help="Update interval in seconds")
    parser.add_argument("--list", "-l", action="store_true", 
                       help="List available experiment logs")
    
    args = parser.parse_args()
    
    monitor = ExperimentLogMonitor(args.log_dir)
    
    if args.list:
        print(f"Available experiment logs in {args.log_dir}:")
        if os.path.exists(args.log_dir):
            for filename in sorted(os.listdir(args.log_dir)):
                if filename.endswith('_main.log'):
                    filepath = os.path.join(args.log_dir, filename)
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    print(f"  {filename} (modified: {mtime.isoformat()})")
        else:
            print(f"  Directory {args.log_dir} does not exist")
        return
    
    monitor.monitor_experiment(args.log_file, args.interval)


if __name__ == "__main__":
    main()