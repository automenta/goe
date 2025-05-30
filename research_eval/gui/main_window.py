import sys
import json
import os
import subprocess # Keep for reference, but QProcess is used
import tempfile
import pandas as pd
import copy # For deepcopy in suggest tab

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QLabel, QMenuBar, QStatusBar, QPushButton, QComboBox, QLineEdit,
    QTextEdit, QFileDialog, QMessageBox, QGridLayout, QHBoxLayout,
    QSizePolicy, QSpinBox, QTableView, QSplitter, QAbstractItemView,
    QFormLayout, QListWidget, QListWidgetItem, QDoubleSpinBox # Added QListWidget, QListWidgetItem, QDoubleSpinBox
)
from PyQt6.QtGui import QAction, QStandardItemModel, QStandardItem
from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

MAIN_SCRIPT_PATH = os.path.join("research_eval", "main.py")
if not os.path.exists(MAIN_SCRIPT_PATH) and os.path.exists(os.path.join("..", MAIN_SCRIPT_PATH)):
     MAIN_SCRIPT_PATH = os.path.join("..", MAIN_SCRIPT_PATH)
elif not os.path.exists(MAIN_SCRIPT_PATH) and os.path.exists(os.path.join("..","..", MAIN_SCRIPT_PATH)):
     MAIN_SCRIPT_PATH = os.path.join("..","..", MAIN_SCRIPT_PATH)
elif not os.path.exists(MAIN_SCRIPT_PATH):
     MAIN_SCRIPT_PATH = "research_eval/main.py"

try:
    from research_eval.db_utils import DB_FILE
except ImportError:
    DB_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "experiment_results.db")
    DB_FILE = os.path.abspath(DB_FILE)
    if not os.path.exists(DB_FILE): DB_FILE = "experiment_results.db"

from research_eval.analyze_results import generate_summary_table, plot_f1_vs_params, plot_f1_vs_latency
ANALYSIS_DB_FILE = DB_FILE

# Imports for Suggest Configurations Tab
from research_eval.suggest_configs import get_top_configurations, generate_variations
SUGGEST_DB_FILE = DB_FILE # Assuming same DB for suggestions


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Research Experimentation Framework")
        self.setGeometry(100, 100, 1200, 800)
        self._create_menu_bar()
        self.setStatusBar(QStatusBar(self))
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)
        self.current_db_path = DB_FILE
        if not self._setup_results_db_connection(self.current_db_path): pass
        self._create_tabs()
        self.experiment_process = None
        self.hpo_process = None
        self.plot_canvas = None
        self.statusBar().showMessage("Ready")

    def _setup_results_db_connection(self, db_path_to_use): # Unchanged
        connection_name = "results_explorer_connection";
        if QSqlDatabase.contains(connection_name): self.db = QSqlDatabase.database(connection_name)
        else: self.db = QSqlDatabase.addDatabase("QSQLITE", connection_name)
        self.db.setDatabaseName(db_path_to_use)
        if not self.db.open():
            QMessageBox.critical(self, "DB Error", f"Could not open DB: {self.db.lastError().text()}\nPath: {db_path_to_use}");
            self.statusBar().showMessage(f"Error: DB connection failed for {db_path_to_use}."); return False
        self.statusBar().showMessage(f"Connected to DB: {db_path_to_use}"); return True

    def _create_menu_bar(self): # Unchanged
        menu_bar=self.menuBar();file_menu=menu_bar.addMenu("&File");exit_action=QAction("&Exit",self);exit_action.triggered.connect(self.close);exit_action.setShortcut("Ctrl+Q");file_menu.addAction(exit_action)
        help_menu=menu_bar.addMenu("&Help");about_action=QAction("&About",self);help_menu.addAction(about_action)

    def _create_tabs(self): # Unchanged
        self._create_experiment_config_tab(); self._create_hpo_setup_tab(); self._create_results_display_tab();
        self._create_analysis_tab(); self._create_suggest_tab()

    def _create_experiment_config_tab(self): # Unchanged (minor reformat for length)
        self.exp_config_tab=QWidget();self.tab_widget.addTab(self.exp_config_tab,"Experiment Configuration");layout=QGridLayout(self.exp_config_tab)
        file_ops_layout=QHBoxLayout();self.load_json_button=QPushButton("Load JSON Config");self.load_json_button.clicked.connect(self._load_json_config);file_ops_layout.addWidget(self.load_json_button)
        self.save_json_button=QPushButton("Save JSON Config");self.save_json_button.clicked.connect(self._save_json_config);file_ops_layout.addWidget(self.save_json_button);layout.addLayout(file_ops_layout,0,0,1,2)
        layout.addWidget(QLabel("Model Type:"),1,0);self.model_type_combo=QComboBox();self.model_type_combo.addItems(["dense","moe","goe","goe_original"]);layout.addWidget(self.model_type_combo,1,1)
        layout.addWidget(QLabel("Dataset Type:"),2,0);self.dataset_type_combo=QComboBox();self.dataset_type_combo.addItems(["synthetic","real_world"]);layout.addWidget(self.dataset_type_combo,2,1)
        layout.addWidget(QLabel("Dataset Name:"),3,0);self.dataset_name_edit=QLineEdit();layout.addWidget(self.dataset_name_edit,3,1)
        layout.addWidget(QLabel("Full JSON Configuration:"),4,0,1,2);self.json_config_edit=QTextEdit();self.json_config_edit.setPlaceholderText("Enter JSON or load.");self.json_config_edit.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);layout.addWidget(self.json_config_edit,5,0,1,2)
        self.run_experiment_button=QPushButton("Run Experiment");self.run_experiment_button.clicked.connect(self._run_experiment);layout.addWidget(self.run_experiment_button,6,0,1,2)
        layout.addWidget(QLabel("Experiment Output:"),7,0,1,2);self.experiment_output_edit=QTextEdit();self.experiment_output_edit.setReadOnly(True);self.experiment_output_edit.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);layout.addWidget(self.experiment_output_edit,8,0,1,2)
        layout.setRowStretch(5,3);layout.setRowStretch(8,2);layout.setColumnStretch(1,1);self._populate_initial_config_template()

    def _populate_initial_config_template(self): # Unchanged
        template={"experiment_name":"my_exp_001","model_type":"dense","dataset_type":"synthetic","dataset_name":"parity","epochs":10,"batch_size":32,
        "model_params":{"embed_dim":64,"num_heads":4,"num_layers":3,"dim_feedforward_factor":4,"dropout":0.1},
        "dataset_params":{"seq_len":20,"vocab_size":10},"optimizer_params":{"name":"adamw","lr":0.001,"weight_decay":0.01},
        "scheduler_params":{"name":"cosine_warmup","num_warmup_steps_factor":0.1,"num_training_steps_factor":1.0},
        "fixed_args":{"grad_clip_norm":1.0,"aux_loss_coeff":0.01}};self.json_config_edit.setText(json.dumps(template,indent=2));
        self.model_type_combo.setCurrentText(template.get("model_type","dense"));self.dataset_type_combo.setCurrentText(template.get("dataset_type","synthetic"));self.dataset_name_edit.setText(template.get("dataset_name","parity"))

    def _load_json_config_from_path(self, filepath): # New helper method
        try:
            with open(filepath, 'r') as f: content = f.read()
            config_data = json.loads(content)
            self.model_type_combo.setCurrentText(config_data.get("model_type", ""))
            self.dataset_type_combo.setCurrentText(config_data.get("dataset_type", ""))
            self.dataset_name_edit.setText(config_data.get("dataset_name", ""))
            self.json_config_edit.setText(json.dumps(config_data, indent=2))
            self.statusBar().showMessage(f"Loaded configuration from {filepath}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load file: {filepath}\nError: {e}")
            self.statusBar().showMessage(f"Error loading {filepath}")
            return False

    def _load_json_config(self): # Modified to use helper
        file_name, _ = QFileDialog.getOpenFileName(self, "Load JSON Configuration", "", "JSON Files (*.json)")
        if file_name: self._load_json_config_from_path(file_name)

    def _save_json_config(self): # Unchanged
        f_name, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON (*.json)");
        if f_name:
            try: config_data=json.loads(self.json_config_edit.toPlainText());config_data["model_type"]=self.model_type_combo.currentText();config_data["dataset_type"]=self.dataset_type_combo.currentText();config_data["dataset_name"]=self.dataset_name_edit.text()
                with open(f_name,'w') as f: json.dump(config_data,f,indent=2);self.statusBar().showMessage(f"Saved to {f_name}")
            except Exception as e: QMessageBox.critical(self,"Save Error",str(e))

    def _run_experiment(self): # Unchanged
        self.experiment_output_edit.clear();self.statusBar().showMessage("Running...");self.run_experiment_button.setEnabled(False)
        try: config=json.loads(self.json_config_edit.toPlainText());config["model_type"]=self.model_type_combo.currentText();config["dataset_type"]=self.dataset_type_combo.currentText();config["dataset_name"]=self.dataset_name_edit.text()
            if "experiment_name" not in config: config["experiment_name"]=f"{config['model_type']}_{config['dataset_name']}_run"
        except json.JSONDecodeError as e: QMessageBox.critical(self,"Run Error",f"Invalid JSON: {e}");self.statusBar().showMessage("Error: Invalid JSON.");self.run_experiment_button.setEnabled(True);return
        try:
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as tmp_f:json.dump(config,tmp_f);self.temp_config_path=tmp_f.name
            self.experiment_process=QProcess(self);self.experiment_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.experiment_process.readyReadStandardOutput.connect(self._handle_experiment_output);self.experiment_process.finished.connect(self._handle_experiment_finished)
            self.experiment_process.start("python",[MAIN_SCRIPT_PATH,"--json_config_path",self.temp_config_path])
        except Exception as e:QMessageBox.critical(self,"Run Error",f"Failed to start: {e}");self.statusBar().showMessage(f"Error: {e}");self.run_experiment_button.setEnabled(True)
            if hasattr(self,'temp_config_path') and os.path.exists(self.temp_config_path):os.unlink(self.temp_config_path)

    def _handle_experiment_output(self): # Unchanged
        if self.experiment_process: self.experiment_output_edit.append(self.experiment_process.readAllStandardOutput().data().decode())
    def _handle_experiment_finished(self): # Unchanged
        proc,btn,out_edit=self.experiment_process,self.run_experiment_button,self.experiment_output_edit;temp_path=getattr(self,'temp_config_path',None)
        if proc:msg="success" if proc.exitStatus()==QProcess.ExitStatus.NormalExit and proc.exitCode()==0 else f"errors (Code: {proc.exitCode()})";self.statusBar().showMessage(f"Experiment {msg}");out_edit.append(f"\n--- Experiment {msg} ---");btn.setEnabled(True)
            if temp_path and os.path.exists(temp_path):
                try:os.unlink(temp_path)
                except OSError as e:out_edit.append(f"\nError deleting temp file: {e}")
            self.experiment_process=None

    def _create_hpo_setup_tab(self): # Unchanged (minor reformat)
        self.hpo_setup_tab=QWidget();self.tab_widget.addTab(self.hpo_setup_tab,"Hyperparameter Optimization");layout=QGridLayout(self.hpo_setup_tab)
        layout.addWidget(QLabel("Study Name:"),0,0);self.hpo_study_name_edit=QLineEdit("my_hpo_study");layout.addWidget(self.hpo_study_name_edit,0,1)
        layout.addWidget(QLabel("Trials:"),1,0);self.hpo_n_trials_spinbox=QSpinBox();self.hpo_n_trials_spinbox.setRange(1,10000);self.hpo_n_trials_spinbox.setValue(100);layout.addWidget(self.hpo_n_trials_spinbox,1,1)
        layout.addWidget(QLabel("Storage URL:"),2,0);self.hpo_storage_edit=QLineEdit("sqlite:///hpo_study.db");layout.addWidget(self.hpo_storage_edit,2,1)
        layout.addWidget(QLabel("Model Type to Optimize:"),3,0);self.hpo_model_type_combo=QComboBox();self.hpo_model_type_combo.addItems(["All-Internal-Choice","dense","moe","goe","goe_original"]);layout.addWidget(self.hpo_model_type_combo,3,1)
        layout.addWidget(QLabel("Dataset Type:"),4,0);self.hpo_dataset_type_combo=QComboBox();self.hpo_dataset_type_combo.addItems(["synthetic","real_world"]);self.hpo_dataset_type_combo.currentIndexChanged.connect(self._update_hpo_dataset_params_visibility);layout.addWidget(self.hpo_dataset_type_combo,4,1)
        layout.addWidget(QLabel("Dataset Name:"),5,0);self.hpo_dataset_name_edit=QLineEdit("parity");layout.addWidget(self.hpo_dataset_name_edit,5,1)
        layout.addWidget(QLabel("Epochs:"),6,0);self.hpo_epochs_spinbox=QSpinBox();self.hpo_epochs_spinbox.setRange(1,1000);self.hpo_epochs_spinbox.setValue(10);layout.addWidget(self.hpo_epochs_spinbox,6,1)
        layout.addWidget(QLabel("Batch Size:"),7,0);self.hpo_batch_size_spinbox=QSpinBox();self.hpo_batch_size_spinbox.setRange(1,1024);self.hpo_batch_size_spinbox.setValue(32);layout.addWidget(self.hpo_batch_size_spinbox,7,1)
        self.hpo_syn_num_samples_label=QLabel("Synth Num Samples:");layout.addWidget(self.hpo_syn_num_samples_label,8,0);self.hpo_syn_num_samples_spinbox=QSpinBox();self.hpo_syn_num_samples_spinbox.setRange(100,100000);self.hpo_syn_num_samples_spinbox.setValue(1000);layout.addWidget(self.hpo_syn_num_samples_spinbox,8,1)
        self.hpo_rw_max_length_label=QLabel("Real Max Length:");layout.addWidget(self.hpo_rw_max_length_label,9,0);self.hpo_rw_max_length_spinbox=QSpinBox();self.hpo_rw_max_length_spinbox.setRange(16,1024);self.hpo_rw_max_length_spinbox.setValue(128);layout.addWidget(self.hpo_rw_max_length_spinbox,9,1)
        self._update_hpo_dataset_params_visibility()
        self.run_hpo_button=QPushButton("Run HPO Study");self.run_hpo_button.clicked.connect(self._run_hpo_study);layout.addWidget(self.run_hpo_button,10,0,1,2)
        layout.addWidget(QLabel("HPO Output:"),11,0,1,2);self.hpo_output_edit=QTextEdit();self.hpo_output_edit.setReadOnly(True);self.hpo_output_edit.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);layout.addWidget(self.hpo_output_edit,12,0,1,2)
        layout.setRowStretch(12,1);layout.setColumnStretch(1,1)

    def _update_hpo_dataset_params_visibility(self): # Unchanged
        is_synthetic = self.hpo_dataset_type_combo.currentText() == "synthetic"
        self.hpo_syn_num_samples_label.setVisible(is_synthetic); self.hpo_syn_num_samples_spinbox.setVisible(is_synthetic)
        self.hpo_rw_max_length_label.setVisible(not is_synthetic); self.hpo_rw_max_length_spinbox.setVisible(not is_synthetic)
    def _run_hpo_study(self): # Unchanged
        self.hpo_output_edit.clear();self.statusBar().showMessage("Running HPO...");self.run_hpo_button.setEnabled(False)
        cmd=["python",MAIN_SCRIPT_PATH,"--hpo_mode"];cmd.extend(["--study_name",self.hpo_study_name_edit.text()]);cmd.extend(["--storage",self.hpo_storage_edit.text()]);cmd.extend(["--n_trials",str(self.hpo_n_trials_spinbox.value())])
        sel_model=self.hpo_model_type_combo.currentText();
        if sel_model!="All-Internal-Choice":cmd.extend(["--hpo_fixed_model_type",sel_model])
        cmd.extend(["--dataset_type",self.hpo_dataset_type_combo.currentText()]);cmd.extend(["--dataset_name",self.hpo_dataset_name_edit.text()])
        cmd.extend(["--epochs",str(self.hpo_epochs_spinbox.value())]);cmd.extend(["--batch_size",str(self.hpo_batch_size_spinbox.value())])
        if self.hpo_dataset_type_combo.currentText()=="synthetic":cmd.extend(["--syn_num_samples",str(self.hpo_syn_num_samples_spinbox.value())])
        else:cmd.extend(["--rw_max_length",str(self.hpo_rw_max_length_spinbox.value())])
        try:self.hpo_process=QProcess(self);self.hpo_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.hpo_process.readyReadStandardOutput.connect(self._handle_hpo_output);self.hpo_process.finished.connect(self._handle_hpo_finished)
            env=QProcessEnvironment.systemEnvironment();self.hpo_process.setProcessEnvironment(env);self.hpo_process.start(cmd[0],cmd[1:]);self.hpo_output_edit.append(f"Starting HPO: {' '.join(cmd)}\n")
        except Exception as e:QMessageBox.critical(self,"HPO Error",f"Failed to start HPO: {e}");self.statusBar().showMessage(f"Error: {e}");self.run_hpo_button.setEnabled(True)
    def _handle_hpo_output(self): # Unchanged
        if self.hpo_process: self.hpo_output_edit.append(self.hpo_process.readAllStandardOutput().data().decode())
    def _handle_hpo_finished(self): # Unchanged
        proc,btn,out_edit=self.hpo_process,self.run_hpo_button,self.hpo_output_edit
        if proc:msg="success" if proc.exitStatus()==QProcess.ExitStatus.NormalExit and proc.exitCode()==0 else f"errors (Code: {proc.exitCode()})";self.statusBar().showMessage(f"HPO {msg}");out_edit.append(f"\n--- HPO {msg} ---");btn.setEnabled(True);self.hpo_process=None

    def _create_results_display_tab(self): # Unchanged
        self.results_display_tab=QWidget();self.tab_widget.addTab(self.results_display_tab,"Results Explorer");main_layout=QVBoxLayout(self.results_display_tab)
        self.refresh_results_button=QPushButton("Refresh Data");self.refresh_results_button.clicked.connect(self._refresh_results_views);main_layout.addWidget(self.refresh_results_button)
        splitter=QSplitter(Qt.Orientation.Vertical);experiments_widget=QWidget();experiments_layout=QVBoxLayout(experiments_widget)
        experiments_layout.addWidget(QLabel("Experiments"));self.experiments_table_view=QTableView()
        self.experiments_table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);self.experiments_table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.experiments_table_view.setSortingEnabled(True);self.experiments_table_view.setAlternatingRowColors(True);experiments_layout.addWidget(self.experiments_table_view);splitter.addWidget(experiments_widget)
        metrics_widget=QWidget();metrics_layout=QVBoxLayout(metrics_widget);metrics_layout.addWidget(QLabel("Metrics for Selected Experiment"));self.metrics_table_view=QTableView()
        self.metrics_table_view.setSortingEnabled(True);self.metrics_table_view.setAlternatingRowColors(True);metrics_layout.addWidget(self.metrics_table_view);splitter.addWidget(metrics_widget)
        main_layout.addWidget(splitter);self._setup_experiments_table_model();self._setup_metrics_table_model()
    def _setup_experiments_table_model(self): # Unchanged
        if not self.db.isOpen():self.statusBar().showMessage("DB not open for exp table.");return
        self.experiments_model=QSqlTableModel(self,db=self.db);self.experiments_model.setTable("experiments");self.experiments_model.setEditStrategy(QSqlTableModel.EditStrategy.OnManualSubmit);self.experiments_model.select()
        self.experiments_table_view.setModel(self.experiments_model);self.experiments_table_view.hideColumn(self.experiments_model.fieldIndex("full_config_json"));self.experiments_table_view.hideColumn(self.experiments_model.fieldIndex("env_details_json"));self.experiments_table_view.resizeColumnsToContents()
        sel_model=self.experiments_table_view.selectionModel();
        if sel_model:sel_model.selectionChanged.connect(self._on_experiment_selected)
        else:self.statusBar().showMessage("Error: No selection model for exp table.")
    def _setup_metrics_table_model(self): # Unchanged
        if not self.db.isOpen():self.statusBar().showMessage("DB not open for metrics table.");return
        self.metrics_model=QSqlTableModel(self,db=self.db);self.metrics_model.setTable("metrics");self.metrics_model.setEditStrategy(QSqlTableModel.EditStrategy.OnManualSubmit)
        self.metrics_model.setFilter("experiment_id = 'INVALID_ID_ON_INIT'");self.metrics_model.select();self.metrics_table_view.setModel(self.metrics_model);self.metrics_table_view.resizeColumnsToContents()
    def _on_experiment_selected(self): # Unchanged
        sel_model=self.experiments_table_view.selectionModel()
        if not sel_model or not sel_model.hasSelection():self.metrics_model.setFilter("experiment_id = 'INVALID_ID_NO_SELECTION'")
        else:
            sel_rows=sel_model.selectedRows()
            if sel_rows:record=self.experiments_model.record(sel_rows[0].row());exp_id=record.value("experiment_id")
                if exp_id is not None:self.metrics_model.setFilter(f"experiment_id = '{exp_id}'")
                else:self.metrics_model.setFilter("experiment_id = 'INVALID_ID_NULL_EXP_ID'")
            else:self.metrics_model.setFilter("experiment_id = 'INVALID_ID_NO_ROWS_ARRAY'")
        self.metrics_model.select();self.metrics_table_view.resizeColumnsToContents()
    def _refresh_results_views(self): # Unchanged
        if hasattr(self,'experiments_model')and self.experiments_model:self.experiments_model.select();self._on_experiment_selected();self.statusBar().showMessage("Results refreshed.")
        else:self.statusBar().showMessage("Results models not init.")

    def _create_analysis_tab(self): # Unchanged (minor reformat)
        self.analysis_tab=QWidget();self.tab_widget.addTab(self.analysis_tab,"Analysis Tools");main_layout=QVBoxLayout(self.analysis_tab)
        summary_group_label=QLabel("Generate Experiment Summary Table");main_layout.addWidget(summary_group_label)
        summary_group_layout=QFormLayout();self.analysis_dataset_name_edit=QLineEdit();summary_group_layout.addRow(QLabel("Dataset Name:"),self.analysis_dataset_name_edit)
        self.analysis_metric_combo=QComboBox();self.analysis_metric_combo.addItems(["final_val_f1","final_val_acc","param_count","inference_latency_ms_batch"]);summary_group_layout.addRow(QLabel("Metric to Optimize:"),self.analysis_metric_combo)
        self.analysis_model_types_edit=QLineEdit();self.analysis_model_types_edit.setPlaceholderText("Optional, e.g., dense,moe");summary_group_layout.addRow(QLabel("Model Types (comma-sep):"),self.analysis_model_types_edit)
        main_layout.addLayout(summary_group_layout)
        self.generate_summary_button=QPushButton("Generate Summary Table");self.generate_summary_button.clicked.connect(self._generate_and_show_summary_table);main_layout.addWidget(self.generate_summary_button)
        self.summary_table_view=QTableView();self.summary_table_view.setSortingEnabled(True);self.summary_table_view.setAlternatingRowColors(True);main_layout.addWidget(self.summary_table_view)
        plot_controls_layout=QHBoxLayout();self.generate_f1_vs_params_button=QPushButton("Generate F1 vs. Params Plot");self.generate_f1_vs_params_button.clicked.connect(self._generate_f1_vs_params_plot);plot_controls_layout.addWidget(self.generate_f1_vs_params_button)
        self.generate_f1_vs_latency_button=QPushButton("Generate F1 vs. Latency Plot");self.generate_f1_vs_latency_button.clicked.connect(self._generate_f1_vs_latency_plot);plot_controls_layout.addWidget(self.generate_f1_vs_latency_button)
        main_layout.addLayout(plot_controls_layout)
        self.plot_display_layout=QVBoxLayout();main_layout.addLayout(self.plot_display_layout)
        main_layout.setStretchFactor(self.summary_table_view,1);main_layout.setStretchFactor(self.plot_display_layout,2)
    def analysis_temp_output_dir(self): # Unchanged
        path=os.path.join(tempfile.gettempdir(),"research_eval_gui_analysis");os.makedirs(path,exist_ok=True);return path
    def _generate_and_show_summary_table(self): # Unchanged
        dataset_name=self.analysis_dataset_name_edit.text().strip()
        if not dataset_name:QMessageBox.warning(self,"Input Error","Dataset Name cannot be empty.");return
        metric_to_optimize=self.analysis_metric_combo.currentText();model_types_str=self.analysis_model_types_edit.text().strip()
        model_types_list=[mt.strip()for mt in model_types_str.split(',')]if model_types_str else None;temp_dir=self.analysis_temp_output_dir()
        try:
            summary_df=generate_summary_table(db_file=ANALYSIS_DB_FILE,dataset_name=dataset_name,model_types=model_types_list,metric_to_optimize=metric_to_optimize,output_dir=temp_dir)
            if not summary_df.empty:self._display_df_in_table_view(summary_df,self.summary_table_view);csv_filename=f"{dataset_name}_summary_table_best_{metric_to_optimize.replace('_','')}.csv";QMessageBox.information(self,"Summary Generated",f"Summary table generated.\nCSV saved to: {os.path.join(temp_dir,csv_filename)}")
            else:QMessageBox.warning(self,"No Data","No data for summary.");self._display_df_in_table_view(pd.DataFrame(),self.summary_table_view)
        except Exception as e:QMessageBox.critical(self,"Error",f"Failed to generate summary: {e}");self._display_df_in_table_view(pd.DataFrame(),self.summary_table_view)
    def _display_df_in_table_view(self,df:pd.DataFrame,table_view:QTableView): # Unchanged
        model=QStandardItemModel(df.shape[0],df.shape[1],self);model.setHorizontalHeaderLabels(df.columns.tolist())
        for r_idx,row in enumerate(df.values):
            for c_idx,value in enumerate(row):model.setItem(r_idx,c_idx,QStandardItem(str(value)))
        table_view.setModel(model);table_view.resizeColumnsToContents();table_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    def _generate_f1_vs_params_plot(self): # Unchanged
        dataset_name=self.analysis_dataset_name_edit.text().strip();
        if not dataset_name:QMessageBox.warning(self,"Input Error","Dataset Name for plot cannot be empty.");return
        model_types_str=self.analysis_model_types_edit.text().strip();model_types_list=[mt.strip()for mt in model_types_str.split(',')]if model_types_str else None
        temp_dir=self.analysis_temp_output_dir()
        try:
            fig=plot_f1_vs_params(db_file=ANALYSIS_DB_FILE,dataset_name=dataset_name,model_types=model_types_list,output_dir=temp_dir,output_image_name=f"{dataset_name}_f1_vs_params_gui.png")
            if fig:self._display_mpl_figure(fig);self.statusBar().showMessage(f"F1 vs. Params plot generated. Saved to {temp_dir}")
            else:QMessageBox.warning(self,"No Data","Not enough data for F1 vs. Params plot.")
        except Exception as e:QMessageBox.critical(self,"Plot Error",f"Failed to generate F1 vs. Params plot: {e}")
    def _generate_f1_vs_latency_plot(self): # Unchanged
        dataset_name=self.analysis_dataset_name_edit.text().strip();
        if not dataset_name:QMessageBox.warning(self,"Input Error","Dataset Name for plot cannot be empty.");return
        model_types_str=self.analysis_model_types_edit.text().strip();model_types_list=[mt.strip()for mt in model_types_str.split(',')]if model_types_str else None
        temp_dir=self.analysis_temp_output_dir()
        try:
            fig=plot_f1_vs_latency(db_file=ANALYSIS_DB_FILE,dataset_name=dataset_name,model_types=model_types_list,output_dir=temp_dir,output_image_name=f"{dataset_name}_f1_vs_latency_gui.png")
            if fig:self._display_mpl_figure(fig);self.statusBar().showMessage(f"F1 vs. Latency plot generated. Saved to {temp_dir}")
            else:QMessageBox.warning(self,"No Data","Not enough data for F1 vs. Latency plot.")
        except Exception as e:QMessageBox.critical(self,"Plot Error",f"Failed to generate F1 vs. Latency plot: {e}")
    def _display_mpl_figure(self,fig:Figure): # Unchanged
        if self.plot_canvas:self.plot_display_layout.removeWidget(self.plot_canvas);self.plot_canvas.deleteLater();self.plot_canvas=None
        self.plot_canvas=FigureCanvas(fig);self.plot_display_layout.addWidget(self.plot_canvas);self.plot_canvas.draw()

    def _create_suggest_tab(self):
        self.suggest_tab = QWidget()
        self.tab_widget.addTab(self.suggest_tab, "Suggest Configurations")
        main_layout = QVBoxLayout(self.suggest_tab)

        # Input Parameters Section
        input_form_layout = QFormLayout()

        self.suggest_dataset_name_edit = QLineEdit()
        input_form_layout.addRow(QLabel("Dataset Name:"), self.suggest_dataset_name_edit)

        self.suggest_model_type_combo = QComboBox()
        self.suggest_model_type_combo.addItems(["dense", "moe", "goe", "goe_original"]) # Common models
        input_form_layout.addRow(QLabel("Model Type:"), self.suggest_model_type_combo)

        self.suggest_metric_combo = QComboBox()
        self.suggest_metric_combo.addItems(["final_val_f1", "final_val_acc", "param_count", "inference_latency_ms_batch"])
        input_form_layout.addRow(QLabel("Metric to Optimize:"), self.suggest_metric_combo)

        self.suggest_top_n_spinbox = QSpinBox()
        self.suggest_top_n_spinbox.setRange(1, 100)
        self.suggest_top_n_spinbox.setValue(3)
        input_form_layout.addRow(QLabel("Top N Configurations:"), self.suggest_top_n_spinbox)

        self.suggest_variations_spinbox = QSpinBox()
        self.suggest_variations_spinbox.setRange(1, 10)
        self.suggest_variations_spinbox.setValue(2)
        input_form_layout.addRow(QLabel("Variations per Config:"), self.suggest_variations_spinbox)

        self.suggest_perturb_mag_spinbox = QDoubleSpinBox()
        self.suggest_perturb_mag_spinbox.setRange(0.01, 1.0)
        self.suggest_perturb_mag_spinbox.setDecimals(2)
        self.suggest_perturb_mag_spinbox.setSingleStep(0.01)
        self.suggest_perturb_mag_spinbox.setValue(0.10)
        input_form_layout.addRow(QLabel("Perturbation Magnitude:"), self.suggest_perturb_mag_spinbox)

        main_layout.addLayout(input_form_layout)

        self.suggest_configs_button = QPushButton("Suggest New Configurations")
        self.suggest_configs_button.clicked.connect(self._suggest_new_configurations)
        main_layout.addWidget(self.suggest_configs_button)

        # Output Section
        main_layout.addWidget(QLabel("Suggested Configuration Files: (Double-click to load into Experiment Tab)"))
        self.suggested_configs_listwidget = QListWidget()
        self.suggested_configs_listwidget.itemDoubleClicked.connect(self._load_selected_suggestion)
        main_layout.addWidget(self.suggested_configs_listwidget)

        main_layout.setStretchFactor(self.suggested_configs_listwidget, 1) # Give list widget stretch

    def suggestions_output_dir(self): # Helper for suggestions output path
        path = os.path.join(tempfile.gettempdir(), "research_eval_gui_suggestions")
        os.makedirs(path, exist_ok=True)
        return path

    def _suggest_new_configurations(self):
        dataset_name = self.suggest_dataset_name_edit.text().strip()
        if not dataset_name:
            QMessageBox.warning(self, "Input Error", "Dataset Name cannot be empty for suggestions.")
            return

        model_type = self.suggest_model_type_combo.currentText()
        metric_name = self.suggest_metric_combo.currentText()
        top_n = self.suggest_top_n_spinbox.value()
        num_variations = self.suggest_variations_spinbox.value()
        perturb_magnitude = self.suggest_perturb_mag_spinbox.value()

        # Define which metrics are "higher is better"
        higher_is_better_metrics = ['final_val_f1', 'final_val_acc']
        higher_is_better = metric_name in higher_is_better_metrics

        try:
            self.statusBar().showMessage(f"Generating suggestions for {dataset_name} - {model_type}...")
            top_configs = get_top_configurations(
                db_file=SUGGEST_DB_FILE, # Assuming SUGGEST_DB_FILE is same as self.current_db_path
                dataset_name=dataset_name,
                model_type=model_type,
                metric_name=metric_name,
                top_n=top_n,
                higher_is_better=higher_is_better
            )

            if not top_configs:
                QMessageBox.warning(self, "No Base Configurations", "No suitable top configurations found in the database to generate variations from.")
                self.suggested_configs_listwidget.clear()
                self.statusBar().showMessage("No base configurations found for suggestions.")
                return

            output_dir = self.suggestions_output_dir()
            self.suggested_configs_listwidget.clear()
            generated_count = 0

            for base_config in top_configs:
                # generate_variations expects a clean config without the _original_ fields
                clean_base_config = {k: v for k, v in base_config.items() if not k.startswith('_original_')}

                variations = generate_variations(
                    clean_base_config,
                    num_variations=num_variations,
                    perturbation_magnitude=perturb_magnitude
                )
                for var_config in variations:
                    new_exp_name = var_config.get("experiment_name", f"suggested_config_{generated_count+1}")
                    output_filename = os.path.join(output_dir, f"{new_exp_name}.json")
                    with open(output_filename, 'w') as f:
                        json.dump(var_config, f, indent=2)

                    list_item = QListWidgetItem(output_filename)
                    self.suggested_configs_listwidget.addItem(list_item)
                    generated_count +=1

            self.statusBar().showMessage(f"Generated {generated_count} new configurations in {output_dir}.")
            QMessageBox.information(self, "Suggestions Generated", f"Successfully generated {generated_count} new configurations.\nThey are saved in: {output_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Suggestion Error", f"Failed to generate suggestions: {e}")
            self.statusBar().showMessage(f"Error generating suggestions: {e}")


    def _load_selected_suggestion(self, item: QListWidgetItem):
        filepath = item.text()
        if self._load_json_config_from_path(filepath): # Use the helper
            self.tab_widget.setCurrentWidget(self.exp_config_tab) # Switch to experiment tab
            self.statusBar().showMessage(f"Loaded suggested config: {os.path.basename(filepath)}")


    def closeEvent(self, event): # Unchanged
        active_procs = [];
        if self.experiment_process and self.experiment_process.state()==QProcess.ProcessState.Running: active_procs.append(("Experiment",self.experiment_process))
        if self.hpo_process and self.hpo_process.state()==QProcess.ProcessState.Running: active_procs.append(("HPO Study",self.hpo_process))
        if active_procs:
            names = " and ".join([n for n,_ in active_procs]);
            reply = QMessageBox.question(self,'Exit',f"{names} running. Exit anyway (terminates process)?", QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes: [p.kill() for _,p in active_procs]; event.accept()
            else: event.ignore()
        else:
            if hasattr(self,'db') and self.db.isOpen(): self.db.close()
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

[end of research_eval/gui/main_window.py]
