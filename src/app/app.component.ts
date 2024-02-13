import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterOutlet } from '@angular/router';
import { HttpClient } from "@angular/common/http";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  tileSelected = true;
  title = "PR's Gen AI";
  constructor(private route: Router) {
  this.tileSelected = true;
  this.route.navigateByUrl("home");
  }
  gotoYAI() {
    this.route.navigate(["transAI"]);
    this.tileSelected = false;
  }

  gotoEmailAI() {
    this.route.navigateByUrl("emailAI");
    this.tileSelected = false;
  }

  gotoFigmaAI() {
    this.route.navigateByUrl("figmaAI");
    this.tileSelected = false;
  }

  gotoSynGenAI() {
    this.route.navigateByUrl("synGenAI");
    this.tileSelected = false;
  }

}
