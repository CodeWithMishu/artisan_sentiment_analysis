from flask import Flask, render_template, jsonify, request, send_file
from pymongo import MongoClient
import json
from datetime import datetime, timedelta
import pandas as pd
from advanced_sentiment_engine import GovernmentGradeArtisanIntelligence
from config import APIConfig, LocationConfig
import os
import threading
import time
from datetime import datetime
import io
import zipfile
import glob
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'hp-government-secret-key')

# Initialize system
intelligence_system = GovernmentGradeArtisanIntelligence()


def _extract_analysis_payload(record):
    """Normalize different storage shapes and return the inner analysis dict.

    Accepts a DB record or result returned by run_district_analysis and
    returns a dict suitable for passing to generate_government_report.
    """
    if not record:
        return None

    # If record is a dict and already the analysis payload
    if isinstance(record, dict) and record.get('total_posts') is not None:
        return record

    # If record has an 'analysis_results' key that is the payload
    if isinstance(record, dict) and record.get('analysis_results'):
        inner = record.get('analysis_results')
        # inner may itself be wrapped
        if isinstance(inner, dict) and inner.get('analysis_results'):
            return inner.get('analysis_results')
        return inner

    # If record is a wrapper with 'data' or similar
    if isinstance(record, dict) and record.get('data'):
        return _extract_analysis_payload(record.get('data'))

    return None


def background_analysis_loop(interval_seconds: int = None):
    """Run periodic district analyses in a background thread.

    This keeps the `sentiment_analysis` collection up to date so the
    dashboard can display near-real-time reports. The interval is
    configurable via ANALYSIS_INTERVAL_SECONDS env var.
    """
    if interval_seconds is None:
        try:
            interval_seconds = int(os.environ.get('ANALYSIS_INTERVAL_SECONDS', '300'))
        except Exception:
            interval_seconds = 300

    app.logger.info(f"Starting background analysis loop every {interval_seconds}s")

    while True:
        try:
            for district in LocationConfig.HP_DISTRICTS.keys():
                app.logger.info(f"Background analysis: running for {district}")
                result = intelligence_system.run_district_analysis(district)
                # Prefer storing the inner analysis_results payload (compatibility)
                stored_analysis = result.get('analysis_results') if isinstance(result, dict) and result.get('analysis_results') else result
                # Save to DB with analysis_date for realtime endpoint
                try:
                    intelligence_system.sentiment_analysis.replace_one(
                        {'district': district},
                        {
                            'district': district,
                            'analysis_date': datetime.now(),
                            'analysis_results': stored_analysis
                        },
                        upsert=True
                    )
                except Exception as e:
                    app.logger.warning(f"Failed to write analysis result for {district}: {e}")
        except Exception as e:
            app.logger.error(f"Background analysis loop error: {e}")

        time.sleep(interval_seconds)


# Start background analysis thread when app runs
def start_background_thread():
    t = threading.Thread(target=background_analysis_loop, daemon=True)
    t.start()

# Start background worker
start_background_thread()

@app.route('/')
def government_dashboard():
    return render_template('government_dashboard.html')

@app.route('/api/districts')
def get_districts():
    """Get all HP districts with their specialties"""
    return jsonify(LocationConfig.HP_DISTRICTS)

@app.route('/api/analyze/<district>')
def analyze_district(district):
    """Trigger analysis for specific district"""
    try:
        result = intelligence_system.run_district_analysis(district)
        # Prefer storing the inner analysis_results payload (compatibility)
        stored_analysis = result.get('analysis_results') if isinstance(result, dict) and result.get('analysis_results') else result
        # Save result to database
        intelligence_system.sentiment_analysis.replace_one(
            {'district': district},
            {
                'district': district,
                'analysis_date': datetime.now(),
                'analysis_results': stored_analysis
            },
            upsert=True
        )
        return jsonify({
            'status': 'success',
            'message': f'Analysis completed for {district}',
            'data': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/ingest', methods=['POST'])
def ingest_posts():
    """Ingest one or more social posts (JSON) and trigger immediate analysis.

    Expected JSON body: a single post dict or a list of post dicts. Each post should
    include at least: 'district' and 'text'. Optional: 'timestamp', 'category', 'engagement'.
    This endpoint is intended as a free, push-based realtime ingestion path so internal
    teams or lightweight automations (IFTTT/Make/email->webhook/etc.) can POST data.
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'status': 'error', 'message': 'Empty JSON payload'}), 400

        posts = payload if isinstance(payload, list) else [payload]

        inserted_ids = []
        districts_to_update = set()

        for i, post in enumerate(posts):
            # Basic validation
            district = post.get('district')
            text = post.get('text')
            if not district or not text:
                continue

            # Normalize timestamps
            if 'timestamp' in post and isinstance(post['timestamp'], str):
                try:
                    post['timestamp'] = datetime.fromisoformat(post['timestamp'])
                except Exception:
                    post['timestamp'] = datetime.now()
            else:
                post.setdefault('timestamp', datetime.now())

            # Store raw post (upsert by a generated id)
            try:
                # attempt to use replace_one for idempotency if '_id' provided
                if post.get('_id'):
                    intelligence_system.social_posts.replace_one({'_id': post['_id']}, post, upsert=True)
                    inserted_ids.append(post['_id'])
                else:
                    res = intelligence_system.social_posts.insert_one({**post, 'collected_at': datetime.now()})
                    inserted_ids.append(res.get('inserted_id'))
            except Exception as e:
                app.logger.warning(f"Failed to persist ingested post: {e}")
                continue

            districts_to_update.add(district)

        # Trigger background analysis for affected districts to keep reports up-to-date
        def _background_update(districts):
            for d in districts:
                try:
                    app.logger.info(f"Realtime ingest: triggering analysis for {d}")
                    result = intelligence_system.run_district_analysis(d)
                    stored_analysis = result.get('analysis_results') if isinstance(result, dict) and result.get('analysis_results') else result
                    intelligence_system.sentiment_analysis.replace_one(
                        {'district': d},
                        {
                            'district': d,
                            'analysis_date': datetime.now(),
                            'analysis_results': stored_analysis
                        },
                        upsert=True
                    )
                except Exception as e:
                    app.logger.error(f"Realtime update failed for {d}: {e}")

        if districts_to_update:
            t = threading.Thread(target=_background_update, args=(list(districts_to_update),), daemon=True)
            t.start()

        return jsonify({'status': 'accepted', 'inserted': len(inserted_ids)}), 202

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/report/<district>')
def get_district_report(district):
    """Get latest analysis report for district"""
    try:
        # Get latest analysis from database
        latest_analysis = intelligence_system.sentiment_analysis.find_one(
            {'district': district},
            sort=[('analysis_date', -1)]
        )
        
        if not latest_analysis:
            return jsonify({'error': 'No analysis found for this district'}), 404
        
        return jsonify(latest_analysis['analysis_results'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report_md/<district>')
def get_district_report_md(district):
    """Return the latest generated report as Markdown text for the district."""
    try:
        latest_analysis = intelligence_system.sentiment_analysis.find_one(
            {'district': district},
            sort=[('analysis_date', -1)]
        )

        # Normalize stored record to inner analysis payload
        analysis = _extract_analysis_payload(latest_analysis)
        if analysis:
            md = intelligence_system.generate_government_report(district, analysis)
            return md, 200, {'Content-Type': 'text/markdown; charset=utf-8'}

        return jsonify({'error': 'No report available for this district'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/realtime-monitoring')
def realtime_monitoring():
    """Get real-time monitoring dashboard data"""
    
    # Get data for all districts
    all_districts_data = {}
    
    for district in LocationConfig.HP_DISTRICTS.keys():
        latest_analysis = intelligence_system.sentiment_analysis.find_one(
            {'district': district},
            sort=[('analysis_date', -1)]
        )
        
        payload = _extract_analysis_payload(latest_analysis)
        if payload:
            all_districts_data[district] = {
                'last_updated': latest_analysis.get('analysis_date') if isinstance(latest_analysis, dict) else None,
                'overall_sentiment': payload.get('overall_sentiment', 0),
                'total_posts': payload.get('total_posts', 0),
                'high_demand_categories': len([
                    cat for cat, data in payload.get('category_analysis', {}).items()
                    if data.get('demand_level') == 'HIGH'
                ])
            }
    
    return jsonify({
        'timestamp': datetime.now(),
        'districts': all_districts_data,
        'system_status': 'operational'
    })

@app.route('/api/export-report/<district>')
def export_report(district):
    """Export detailed report as PDF/Excel"""
    try:
        # Get latest analysis
        latest_analysis = intelligence_system.sentiment_analysis.find_one(
            {'district': district},
            sort=[('analysis_date', -1)]
        )
        
        if not latest_analysis:
            return jsonify({'error': 'No analysis found'}), 404
        
        # Generate report
        report_content = intelligence_system.generate_government_report(
            district, latest_analysis['analysis_results']
        )
        
        # Save to file
        filename = f"HP_Report_{district}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return send_file(filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-all', methods=['GET'])
def export_all_reports():
    """Export all reports as a zip. Convert MD to PDF when reportlab is available."""
    try:
        # Find report files
        md_files = glob.glob(os.path.join('reports', 'HP_Artisan_Report_*.md'))
        if not md_files:
            return jsonify({'error': 'No reports found'}), 404

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for md in md_files:
                basename = os.path.basename(md)
                if REPORTLAB_AVAILABLE:
                    # Convert to PDF
                    pdf_buf = io.BytesIO()
                    c = canvas.Canvas(pdf_buf, pagesize=letter)
                    with open(md, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    textobject = c.beginText(40, 750)
                    textobject.setFont('Helvetica', 10)
                    y = 750
                    for line in lines:
                        # Basic wrapping
                        for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
                            textobject.textLine(chunk.rstrip('\n'))
                            y -= 12
                            if y < 60:
                                c.drawText(textobject)
                                c.showPage()
                                textobject = c.beginText(40, 750)
                                textobject.setFont('Helvetica', 10)
                                y = 750
                    c.drawText(textobject)
                    c.save()
                    pdf_buf.seek(0)
                    z.writestr(basename.replace('.md', '.pdf'), pdf_buf.read())
                else:
                    # Include raw MD
                    with open(md, 'rb') as f:
                        z.writestr(basename, f.read())

        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='hp_reports.zip', mimetype='application/zip')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Statewide Scan Endpoint ---
@app.route('/api/statewide-scan', methods=['POST'])
def statewide_scan():
    """Trigger analysis for all districts and return summary/results"""
    try:
        districts = list(LocationConfig.HP_DISTRICTS.keys())
        results = {}
        for district in districts:
            try:
                result = intelligence_system.run_district_analysis(district)
                # Prefer storing the inner analysis_results payload (compatibility)
                stored_analysis = result.get('analysis_results') if isinstance(result, dict) and result.get('analysis_results') else result
                # Save result to database
                intelligence_system.sentiment_analysis.replace_one(
                    {'district': district},
                    {
                        'district': district,
                        'analysis_date': datetime.now(),
                        'analysis_results': stored_analysis
                    },
                    upsert=True
                )
                results[district] = {
                    'status': 'success',
                    'data': result
                }
            except Exception as e:
                results[district] = {
                    'status': 'error',
                    'message': str(e)
                }
        return jsonify({
            'status': 'completed',
            'results': results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
